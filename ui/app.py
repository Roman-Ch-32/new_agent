#!/usr/bin/env python3
import os, sys, json, threading, re, time
from websocket import WebSocketApp
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QTextCursor, QColor, QPalette, QTextOption

WS_URL = os.environ.get("AGENT_WS_URL", "ws://127.0.0.1:8000/ws")


def format_response_text(text: str) -> str:
    """Конвертирует markdown в HTML (PyCharm стиль)"""
    if not text:
        return ""

    html = text

    # 1. Экранируем HTML спецсимволы ПЕРЕД обработкой
    html = (html
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;'))

    # 2. Код блоки
    html = re.sub(
        r'```(\w*)\n(.*?)```',
        lambda m: f'<div style="background: #2b2d30; color: #a9b7c6; '
                  f'padding: 12px; border-radius: 4px; margin: 10px 0; '
                  f'font-family: Consolas, Monaco, monospace; font-size: 0.9em; '
                  f'overflow-x: auto; border: 1px solid #3c3f41;">'
                  f'<div style="color: #cc7832; margin-bottom: 8px; font-size: 0.8em;">{m.group(1) or "code"}</div>'
                  f'<pre style="margin: 0; white-space: pre-wrap;">{m.group(2)}</pre>'
                  f'</div>',
        html,
        flags=re.DOTALL
    )

    # 3. Inline код
    html = re.sub(
        r'`([^`]+)`',
        r'<code style="background: #3c3f41; color: #6a8759; '
        r'padding: 2px 6px; border-radius: 3px; '
        r'font-family: Consolas, Monaco, monospace; font-size: 0.9em;">\1</code>',
        html
    )

    # 4. Заголовки
    html = re.sub(r'^### (.+)$', r'<h3 style="color: #a9b7c6; margin: 15px 0 10px 0; font-size: 1.3em;">\1</h3>', html,
                  flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2 style="color: #a9b7c6; margin: 20px 0 12px 0; font-size: 1.5em;">\1</h2>', html,
                  flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1 style="color: #a9b7c6; margin: 25px 0 15px 0; font-size: 1.7em;">\1</h1>', html,
                  flags=re.MULTILINE)

    # 5. Жирный текст
    html = re.sub(r'\*\*(.+?)\*\*', r'<b style="color: #a9b7c6;">\1</b>', html)

    # 6. Курсив
    html = re.sub(r'\*(.+?)\*', r'<i style="color: #808080;">\1</i>', html)

    # 7. Списки
    html = re.sub(r'^[-•*] (.+)$', r'<li style="margin: 5px 0; color: #a9b7c6;">\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li.*?</li>(?:\n<li.*?</li>)*)', r'<ul style="margin: 10px 0; padding-left: 25px;">\1</ul>', html,
                  flags=re.DOTALL)

    # 8. Переносы строк
    html = html.replace('\n\n', '<br><br>')
    html = html.replace('\n', '<br>')

    html = f'<div style="font-family: Arial, sans-serif; line-height: 1.6; color: #a9b7c6; white-space: normal;">{html}</div>'

    return html


class WebSocketWorker(QObject):
    """WebSocket в отдельном QThread"""
    connected = pyqtSignal()
    disconnected = pyqtSignal(int, str)
    message_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.ws = None
        self._running = False

    def connect(self):
        self._running = True
        try:
            self.ws = WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self.ws.run_forever()
        except Exception as e:
            if self._running:
                self.error_occurred.emit(str(e))

    def disconnect(self):
        self._running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            self.ws = None

    def send(self, message: str):
        if self.ws and self._running:
            try:
                self.ws.send(message)
            except Exception as e:
                self.error_occurred.emit(str(e))

    def _on_open(self, ws):
        if self._running:
            self.connected.emit()

    def _on_message(self, ws, message):
        if self._running and message:
            self.message_received.emit(message)

    def _on_error(self, ws, error):
        if self._running:
            self.error_occurred.emit(str(error) if error else "Unknown error")

    def _on_close(self, ws, close_status_code, close_msg):
        if self._running:
            self.disconnected.emit(
                close_status_code if close_status_code else 0,
                close_msg if close_msg else "Closed"
            )


class UiBridge(QObject):
    append = pyqtSignal(str, str)
    status = pyqtSignal(str)
    append_html = pyqtSignal(str)
    connection_status = pyqtSignal(bool)


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 Мульт-Агент (Qt)")
        self.resize(1024, 768)

        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._setup_pycharm_theme()

        self.status_label = QLabel(" ")
        self.status_label.setStyleSheet("color: #808080; font-style: italic; font-size: 12px;")

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setAcceptRichText(True)
        self.chat.setHtml('<div style="color: #808080; font-style: italic;">Подключено к агенту...</div>')
        self.chat.setFont(QFont("Consolas", 11))
        self.chat.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.chat.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextBrowserInteraction)

        self.input = QTextEdit()
        self.input.setFixedHeight(100)
        self.input.setPlaceholderText("Введите команду... (Enter — отправить, Shift+Enter — новая строка)")
        self.input.installEventFilter(self)
        self.input.setFont(QFont("Consolas", 11))

        self.ws_worker = None
        self.ws_thread = None
        self.connected = False
        self._reconnect_lock = False
        self._should_reconnect = True
        self._closing = False

        btn_send = QPushButton("📤 Отправить")
        btn_clear = QPushButton("🗑️ Очистить")
        btn_reconnect = QPushButton("🔄 Переподключиться")

        self.chk_reasoning = QPushButton("🧠 Рассуждения: Вкл")
        self.chk_reasoning.setCheckable(True)
        self.chk_reasoning.setChecked(True)
        self.chk_reasoning.clicked.connect(self.toggle_reasoning)
        self.show_reasoning = True

        btn_send.clicked.connect(self.send)
        btn_clear.clicked.connect(self.clear_chat)
        btn_reconnect.clicked.connect(self.reconnect)

        row = QHBoxLayout()
        row.addWidget(btn_send)
        row.addWidget(btn_clear)
        row.addWidget(btn_reconnect)
        row.addWidget(self.chk_reasoning)

        layout = QVBoxLayout()

        header = QLabel("🎬 <b>Мульт-Агент</b> — AI помощник для Unreal Engine")
        header.setStyleSheet("font-size: 16px; color: #a9b7c6; padding: 10px; font-weight: bold;")
        layout.addWidget(header)

        self.connection_indicator = QLabel("🔴 Отключено")
        self.connection_indicator.setStyleSheet("color: #ff6b68; font-weight: bold; font-size: 12px; padding: 5px;")
        layout.addWidget(self.connection_indicator)

        layout.addWidget(QLabel("🔗 WebSocket: " + WS_URL))
        layout.addWidget(self.chat)
        layout.addWidget(QLabel("✏️ Сообщение:"))
        layout.addWidget(self.input)
        layout.addLayout(row)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        self.bridge = UiBridge()
        self.bridge.append.connect(self._append)
        self.bridge.status.connect(self.status_label.setText)
        self.bridge.append_html.connect(self._append_html)
        self.bridge.connection_status.connect(self._update_connection_indicator)

        self._connect_ws()

    def _setup_pycharm_theme(self):
        """Настраивает тему PyCharm Darcula с ВИДИМЫМ выделением"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(43, 45, 48))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(169, 183, 198))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(43, 45, 48))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(60, 63, 65))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(169, 183, 198))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(60, 63, 65))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(169, 183, 198))
        # ✅ Цвет выделения (синий когда выделяешь мышкой)
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(33, 66, 131))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

        QApplication.instance().setPalette(dark_palette)

        self.setStyleSheet("""
            QWidget {
                background: #2b2d30;
                color: #a9b7c6;
                font-size: 13px;
            }
            QTextEdit {
                background: #2b2d30;
                color: #a9b7c6;
                border: 1px solid #3c3f41;
                border-radius: 4px;
                padding: 8px;
                /* ✅ ВИДИМОЕ ВЫДЕЛЕНИЕ при селекте мышкой */
                selection-background-color: #214283;
                selection-color: #ffffff;
            }
            QTextEdit:focus {
                border: 1px solid #4ec9b0;
            }
            QPushButton {
                background: #3c3f41;
                color: #a9b7c6;
                border: 1px solid #3c3f41;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background: #4c4f51;
            }
            QPushButton:pressed {
                background: #2c2f31;
            }
            QPushButton:checked {
                background: #214283;
                color: #a9b7c6;
            }
            QLabel {
                color: #a9b7c6;
            }
        """)

    def toggle_reasoning(self):
        self.show_reasoning = self.chk_reasoning.isChecked()
        state = "Вкл" if self.show_reasoning else "Выкл"
        self.chk_reasoning.setText(f"🧠 Рассуждения: {state}")

    def clear_chat(self):
        self.chat.clear()
        self.chat.setHtml('<div style="color: #808080; font-style: italic;">Чат очищен</div>')

    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == event.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False
                self.send()
                return True
        return super().eventFilter(obj, event)

    def _cleanup_ws(self):
        if self.ws_worker:
            try:
                self.ws_worker.disconnect()
            except:
                pass
            try:
                self.ws_worker.connected.disconnect()
                self.ws_worker.disconnected.disconnect()
                self.ws_worker.message_received.disconnect()
                self.ws_worker.error_occurred.disconnect()
            except:
                pass
            self.ws_worker = None

        if self.ws_thread and self.ws_thread.isRunning():
            self.ws_thread.quit()
            self.ws_thread.wait(2000)
        self.ws_thread = None

    def _connect_ws(self):
        if self._closing or not self._should_reconnect:
            return

        if self._reconnect_lock:
            return

        self._reconnect_lock = True
        self.bridge.status.emit("🔌 Подключение к агенту...")

        self._cleanup_ws()

        self.ws_worker = WebSocketWorker(WS_URL)
        self.ws_thread = QThread()

        self.ws_worker.moveToThread(self.ws_thread)

        self.ws_worker.connected.connect(self._on_open)
        self.ws_worker.disconnected.connect(self._on_close)
        self.ws_worker.message_received.connect(self._on_message)
        self.ws_worker.error_occurred.connect(self._on_error)

        self.ws_thread.started.connect(self.ws_worker.connect)
        self.ws_thread.start()

    def _on_open(self):
        if self._closing:
            return

        self.connected = True
        self._reconnect_attempts = 0
        self._reconnect_lock = False

        self._append_html('<div style="color: #57965c; margin: 10px 0;"><br>✅ Подключено к агенту!</div>')
        self.bridge.connection_status.emit(True)

    def _on_error(self, error: str):
        if self._closing:
            return

        self.connected = False
        error_msg = error if error else "Неизвестная ошибка"
        self._append_html(f'<div style="color: #ffc66d; margin: 10px 0;"><br>❌ Ошибка: {error_msg}</div>')
        self.bridge.connection_status.emit(False)

    def _on_close(self, close_status_code: int, close_msg: str):
        if self._closing:
            return

        self.connected = False
        self._reconnect_lock = False

        self._append_html(f'<div style="color: #808080; margin: 10px 0;"><br>🔌 Отключено: {close_msg}</div>')
        self.bridge.connection_status.emit(False)

        if not self._should_reconnect:
            return

        if close_status_code != 1000:
            self._reconnect_attempts += 1

            if self._reconnect_attempts > self._max_reconnect_attempts:
                self._append_html('<div style="color: #ff6b68; margin: 10px 0;"><br>❌ Превышено число попыток</div>')
                self._should_reconnect = False
                return

            delay = min(3 * (2 ** (self._reconnect_attempts - 1)), 60)
            self._append_html(
                f'<div style="color: #ffc66d; margin: 10px 0;"><br>🔄 Переподключение через {delay} сек...</div>')

            from PyQt6.QtCore import QTimer
            QTimer.singleShot(int(delay * 1000), self._do_reconnect)

    def _do_reconnect(self):
        if self._closing or not self._should_reconnect:
            return

        self.bridge.status.emit("🔄 Переподключение...")
        self._connect_ws()

    def reconnect(self):
        if self._closing:
            return

        if self._reconnect_lock:
            return

        self._reconnect_attempts = 0
        self._should_reconnect = True

        self._append_html('<div style="color: #57965c; margin: 10px 0;"><br>🔄 Переподключение...</div>')

        self._cleanup_ws()

        time.sleep(0.5)

        self._reconnect_lock = False
        self._connect_ws()

    def _append(self, who: str, text: str):
        formatted = format_response_text(text)
        separator = '<div style="height: 1px; background: #3c3f41; margin: 15px 0;"></div><br>'

        # ✅ Сообщения БЕЗ фона — только бордер слева
        # Фон появляется ТОЛЬКО при выделении мышкой
        if who == "👤 Вы":
            html = f'''
            {separator}
            <div style="max-width: 80%; margin: 10px 0; padding: 12px; border-radius: 4px; 
                        border-left: 4px solid #57965c; display: inline-block;
                        background: transparent;">
                <div style="color: #57965c; font-weight: bold; margin-bottom: 8px;">{who}</div>
                <div style="color: #a9b7c6;">{formatted}</div>
            </div>
            '''
        else:
            html = f'''
            {separator}
            <div style="max-width: 80%; margin: 10px 0; padding: 12px; border-radius: 4px; 
                        border-left: 4px solid #4ec9b0; display: inline-block;
                        background: transparent;">
                <div style="color: #4ec9b0; font-weight: bold; margin-bottom: 8px;">{who}</div>
                <div style="color: #a9b7c6;">{formatted}</div>
            </div>
            '''

        self._append_html(html)

    def _append_html(self, html: str):
        """Безопасное добавление HTML"""
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html)
        self.chat.setTextCursor(cursor)
        cursor.clearSelection()
        self.chat.setTextCursor(cursor)
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def _update_connection_indicator(self, connected: bool):
        if connected:
            self.connection_indicator.setText("🟢 Подключено")
            self.connection_indicator.setStyleSheet("color: #57965c; font-weight: bold; font-size: 12px; padding: 5px;")
        else:
            self.connection_indicator.setText("🔴 Отключено")
            self.connection_indicator.setStyleSheet("color: #ff6b68; font-weight: bold; font-size: 12px; padding: 5px;")

    def _on_message(self, msg: str):
        try:
            obj = json.loads(msg)
        except Exception:
            obj = {"type": "text", "text": msg}

        t = obj.get("type")

        if t == "reasoning_start":
            self._append_html('<div style="color: #ffc66d; font-style: italic;"><br>🤔 Агент думает...</div>')

        elif t == "reasoning":
            reasoning = obj.get("reasoning", "")
            if reasoning and self.show_reasoning:
                reasoning_html = f'''
                <div style="background: #3d3d2d; border-left: 4px solid #ffc66d; 
                            padding: 12px; margin: 10px 0; border-radius: 4px; max-width: 80%;">
                    <div style="color: #ffc66d; font-weight: bold; margin-bottom: 8px;">🧠 Рассуждения:</div>
                    <div style="color: #cccccc; font-style: italic;">{format_response_text(reasoning)}</div>
                </div>
                '''
                self._append_html(reasoning_html)

        elif t == "reply":
            reply = obj.get("reply", "")
            self._append("🤖 Агент", reply)

        elif t == "error":
            error_msg = obj.get("message", obj.get("error", "Неизвестная ошибка"))
            self._append_html(f'<div style="color: #ff6b68;"><br>❌ Ошибка: {error_msg}</div>')

    def send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.setPlainText("")
        self._append("👤 Вы", text)

        if not self.connected or not self.ws_worker:
            QMessageBox.warning(self, "⚠️ Не подключено", "Нет соединения с сервером.")
            return

        try:
            payload = json.dumps({"message": text}, ensure_ascii=False)
            self.ws_worker.send(payload)
        except Exception as e:
            QMessageBox.critical(self, "❌ Ошибка", f"Не удалось отправить: {e}")

    def closeEvent(self, event):
        self._closing = True
        self._should_reconnect = False
        self._cleanup_ws()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = Window()
    w.show()
    sys.exit(app.exec())