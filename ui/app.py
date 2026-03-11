#!/usr/bin/env python3
import os, sys, json, threading, re
from websocket import WebSocketApp
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal

WS_URL = os.environ.get("AGENT_WS_URL", "ws://127.0.0.1:8000/ws")


class UiBridge(QObject):
    """Thread-safe bridge from WS thread -> Qt GUI thread."""
    append = pyqtSignal(str, str)
    status = pyqtSignal(str)


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 Мульт-Агент (Qt)")
        self.resize(980, 720)

        self.status_label = QLabel(" ")
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        # 🔹 Включаем рендеринг HTML для форматирования
        self.chat.setAcceptRichText(True)

        self.input = QTextEdit()
        self.input.setFixedHeight(110)
        self.input.setPlaceholderText("Введите команду... (Enter — отправить, Shift+Enter — новая строка)")
        self.input.installEventFilter(self)

        self.ws_thread = None

        # Кнопки
        btn_send = QPushButton("📤 Отправить")
        btn_clear = QPushButton("🗑️ Очистить")
        btn_reconnect = QPushButton("🔄 Переподключиться")

        # 🔹 НОВАЯ КНОПКА: переключатель рассуждений
        self.chk_reasoning = QPushButton("🧠 Рассуждения: Вкл")
        self.chk_reasoning.setCheckable(True)
        self.chk_reasoning.setChecked(True)
        self.chk_reasoning.clicked.connect(self.toggle_reasoning)
        self.show_reasoning = True  # состояние по умолчанию

        btn_send.clicked.connect(self.send)
        btn_clear.clicked.connect(lambda: self.chat.setPlainText(""))
        btn_reconnect.clicked.connect(self.reconnect)

        # 🔹 Добавляем кнопку в ряд с остальными
        row = QHBoxLayout()
        row.addWidget(btn_send)
        row.addWidget(btn_clear)
        row.addWidget(btn_reconnect)
        row.addWidget(self.chk_reasoning)  # ← вот сюда

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"🔗 WebSocket: {WS_URL}"))
        layout.addWidget(self.chat)
        layout.addWidget(QLabel("✏️ Сообщение:"))
        layout.addWidget(self.input)
        layout.addLayout(row)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.ws = None
        self.connected = False

        self._reconnect_lock = False
        self._reconnect_thread = None

        self.bridge = UiBridge()
        self.bridge.append.connect(self._append)
        self.bridge.status.connect(self.status_label.setText)

        self._connect_ws()


    def toggle_reasoning(self):
        """Переключатель видимости рассуждений"""
        self.show_reasoning = self.chk_reasoning.isChecked()
        state = "Вкл" if self.show_reasoning else "Выкл"
        self.chk_reasoning.setText(f"🧠 Рассуждения: {state}")

    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == event.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False
                self.send()
                return True
        return super().eventFilter(obj, event)

    def _connect_ws(self):
        self.bridge.status.emit("🔌 Подключение к агенту...")
        self.chat.append("⚙️ Подключение к серверу...")

        try:
            self.ws = WebSocketApp(
                WS_URL,
                on_open=self._on_open,
                on_message=lambda ws, msg: self._on_message(msg),
                on_error=self._on_error,
                on_close=self._on_close,
            )
            t = threading.Thread(target=self.ws.run_forever, daemon=True)
            t.start()
        except Exception as e:
            self.bridge.status.emit(f"❌ Ошибка: {e}")
            self.chat.append(f"❌ Не удалось подключиться: {e}")

    def _on_open(self, ws):
        self.connected = True
        self.bridge.status.emit("✅ Подключено")
        self.chat.append("✅ Подключено к агенту!")

    def _on_error(self, ws, error):
        self.connected = False
        self.bridge.status.emit(f"❌ Ошибка: {error}")
        self.chat.append(f"❌ Ошибка соединения: {error}")

    # ui/app.py
    def _on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        self.bridge.status.emit(f"🔌 Отключено: {close_status_code}")
        self.chat.append(f"🔌 Соединение закрыто: {close_msg}")

        # ✅ Блокировка от двойного reconnect
        if hasattr(self, '_reconnect_lock') and self._reconnect_lock:
            return

        if close_status_code != 1000:
            self._reconnect_lock = True  # ← Ставим блокировку
            self.chat.append("🔄 Попытка переподключения через 3 секунды...")

            t = threading.Thread(target=self._reconnect_delayed, daemon=True)
            t.start()
            # ✅ Сохраняем ссылку на тред для очистки
            self._reconnect_thread = t

    def _reconnect_delayed(self):
        import time
        time.sleep(3)
        self.bridge.status.emit("🔄 Переподключение...")
        self._connect_ws()

    def reconnect(self):
        """Ручная переподключение"""
        self._reconnect_lock = False  # ← Сброс блокировки

        if self.ws:
            self.ws.close()
        self._connect_ws()

    def _append(self, who: str, text: str):
        """Безопасное добавление сообщения с прокруткой"""
        self.chat.append(f"<b>{who}:</b> {text}<br>")
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def _parse_reasoning(self, text: str) -> dict:
        """Извлекает рассуждения из текста с тегами <reasoning>...</reasoning>"""
        reasoning = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL | re.IGNORECASE)
        answer = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)

        return {
            "reasoning": reasoning.group(1).strip() if reasoning else "",
            "answer": answer.group(1).strip() if answer else text,
            "has_reasoning": bool(reasoning)
        }

    def _on_message(self, msg: str):
        try:
            obj = json.loads(msg)
        except Exception:
            obj = {"type": "text", "text": msg}

        t = obj.get("type")

        if t == "status":
            self.bridge.status.emit(obj.get("status", ""))

        elif t == "reply":
            reply = obj.get("reply", "")
            reasoning = obj.get("reasoning", "")
            has_reasoning = obj.get("has_reasoning", False)
            usage = obj.get("usage", {})
            session_usage = obj.get("session_usage", {})

            # 🔹 Показываем рассуждения, если они есть И включены
            if has_reasoning and reasoning and self.show_reasoning:
                self.chat.append(
                    f'<div style="color: #666; font-size: 0.9em; background: #f9f9f9; '
                    f'padding: 8px; border-left: 3px solid #007acc; margin: 5px 0;">'
                    f'🧠 <b>Рассуждения:</b><br>{reasoning}</div>'
                )
            elif has_reasoning and not self.show_reasoning:
                # Индикатор, что рассуждения есть, но скрыты
                self.chat.append(
                    '<span style="color: #999; font-size: 0.85em;">'
                    '[🧠 Рассуждения скрыты — включите кнопку выше]</span><br>'
                )

            # 🔹 Финальный ответ
            self.bridge.append.emit("🤖 Агент", reply)

            # 🔹 Статистика токенов (если есть)
            if usage:
                pt = usage.get("prompt_tokens", 0)
                ct = usage.get("completion_tokens", 0)
                tt = usage.get("total_tokens", 0)
                s_tt = session_usage.get("total_tokens", tt)

                self.chat.append(
                    f'<div style="color: #888; font-size: 0.8em; margin-top: 3px;">'
                    f'📊 Токены: <b>{tt}</b> (prompt={pt}, completion={ct}) | '
                    f'Всего за сессию: <b>{s_tt}</b>'
                    f'</div>'
                )

            self.bridge.status.emit("✅ Готово")

        elif t == "stats":
            usage = obj.get("usage") or {}
            if usage:
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                tt = usage.get("total_tokens")
                self.chat.append(
                    f'<span style="color: #888; font-size: 0.85em;">'
                    f'📊 Статистика: prompt={pt}, completion={ct}, total={tt}</span><br>'
                )

        elif t == "metrics":
            session = obj.get("session")
            msgs = obj.get("messages")
            qpts = obj.get("qdrant_points")
            usage = obj.get("usage") or {}
            tt = usage.get("total_tokens")
            self.chat.append(
                f'<span style="color: #888; font-size: 0.85em;">'
                f'📈 Метрики: session={session} messages={msgs} qdrant_points={qpts} total_tokens={tt}</span><br>'
            )

        elif t == "error":
            self.bridge.append.emit("❌ Ошибка", obj.get("error", ""))
            self.bridge.status.emit("⚠️ Ошибка")

        else:
            self.bridge.append.emit("⚙️ Система", msg)

    def send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.setPlainText("")
        self._append("👤 Вы", text)

        if not self.connected:
            QMessageBox.warning(self, "⚠️ Не подключено", "Нет соединения с сервером. Попробуйте переподключиться.")
            return

        try:
            self.bridge.status.emit("📤 Отправка...")
            payload = json.dumps({"message": text}, ensure_ascii=False)
            if self.ws:
                self.ws.send(payload)
        except Exception as e:
            QMessageBox.critical(self, "❌ Ошибка", f"Не удалось отправить: {e}")
            self.bridge.status.emit("❌ Ошибка отправки")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = Window()
    w.show()
    sys.exit(app.exec())