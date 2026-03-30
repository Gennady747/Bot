import asyncio
import os
import torch
import torch.nn as nn
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
import logging

# ================== НАСТРОЙКИ ==================
BOT_TOKEN = os.getenv("BOT_TOKEN")   # ← Берётся с сервера Bothost.ru

if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN не найден в переменных окружения!")

MODEL_PATH = "lstm_model.pth"

# ================== LSTM-МОДЕЛЬ ==================
class MyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Загрузка модели
device = torch.device("cpu")
model = MyLSTM()

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"✅ Модель LSTM успешно загружена | PyTorch {torch.__version__}")
except Exception as e:
    print(f"⚠️ Не удалось загрузить модель: {e}")
    print(f"Убедитесь, что файл {MODEL_PATH} находится в корне проекта")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "👋 <b>LSTM-бот запущен!</b>\n\n"
        "Отправь команду:\n"
        "<code>/predict 1.2 3.4 5.6 7.8</code> — получить предсказание"
    )

@dp.message(Command("predict"))
async def cmd_predict(message: Message):
    try:
        # Получаем числа после команды
        values = list(map(float, message.text.split()[1:]))
        if len(values) < 1:
            raise ValueError("Нет данных")

        input_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)

        await message.reply(f"✅ <b>Предсказание LSTM:</b> {prediction.item():.4f}")
    except Exception as e:
        await message.reply(f"❌ Ошибка.\nПример: <code>/predict 1.2 3.4 5.6 7.8</code>")

@dp.message(Command("status"))
async def cmd_status(message: Message):
    params = sum(p.numel() for p in model.parameters())
    await message.reply(
        f"✅ <b>Бот работает</b>\n"
        f"PyTorch: {torch.__version__}\n"
        f"Параметры модели: {params:,}"
    )

async def main():
    logging.basicConfig(level=logging.INFO)
    print("🚀 LSTM Telegram Bot запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
