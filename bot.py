import discord
from discord.ext import commands
import aiohttp
import asyncio
import logging
import json
from datetime import datetime, timedelta
from aiohttp import web
import os

# Логи
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='bot.log',
    filemode='a'
)

# Настройки Discord
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Настройки
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
BOT_API_KEY = os.getenv("BOT_API_KEY")
CEREBRAS_API_URL = os.getenv("CEREBRAS_API_URL", "https://api.cerebras.ai/v1/chat/completions")
WARNINGS = {}
MAX_WARNINGS_BEFORE_MUTE = 6
MUTE_DURATION = 600
MAX_WARNINGS_BEFORE_BAN = 10

# Проверка переменных окружения
print("Starting bot... Environment variables:")
print(f"DISCORD_BOT_TOKEN: {'Set' if DISCORD_BOT_TOKEN else 'Not set'}")
print(f"CEREBRAS_API_KEY: {'Set' if CEREBRAS_API_KEY else 'Not set'}")
print(f"BOT_API_KEY: {'Set' if BOT_API_KEY else 'Not set'}")
print(f"CEREBRAS_API_URL: {'Set' if CEREBRAS_API_URL else 'Not set'}")

# Проверка наличия ключей
if not all([CEREBRAS_API_KEY, DISCORD_BOT_TOKEN, BOT_API_KEY]):
    logging.error("Один или несколько ключей не установлены")
    raise Exception("Необходимые ключи (CEREBRAS_API_KEY, DISCORD_BOT_TOKEN, BOT_API_KEY) не установлены")

# Загрузка предупреждений
def load_warnings():
    global WARNINGS
    try:
        with open('warnings.json', 'r') as f:
            WARNINGS = {int(k): v for k, v in json.load(f).items()}
        logging.info("Предупреждения загружены успешно")
    except FileNotFoundError:
        WARNINGS = {}
        logging.info("Файл warnings.json не найден, создан пустой словарь предупреждений")

# Сохранение предупреждений
def save_warnings():
    try:
        with open('warnings.json', 'w') as f:
            json.dump(WARNINGS, f)
        logging.info("Предупреждения сохранены")
    except Exception as e:
        logging.error(f"Ошибка при сохранении предупреждений: {e}")

# Cerebras API
async def get_cerebras_response(prompt, is_retry=False, is_moderation=False):
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }
    system_prompt = (
        "Ты МаксимAI, весёлый и дружелюбный бот! Обожаешь тусить с народом и болтать на любые темы. "
        "Ты предприниматель, который продаёт булочки — твой крутой бизнес, и ты гордишься своим ассортиментом! "
        "Чёрный хлеб — твоя главная любовь, без него ни дня! Отвечай на русском языке, будь дружелюбным, используй сленг и лёгкий юмор про булочки и чёрный хлеб. "
        "Давай развёрнутые ответы (3-4 предложения), чтобы было интересно, но не затягивай. "
        "Если вопрос простой, вроде 'привет', отвечай живо и с душой, например: 'Бро, привет! Дела как мой бизнес с булочками — в гору! А у тебя как? Чёрный хлеб ел сегодня? 😎'. "
        "Если не знаешь ответа, честно признайся и предложи что-нибудь прикольное. Завершай ответы логично, с точкой или восклицательным знаком!"
    ) if not is_moderation else (
        "Ты модератор чата. Оцени, содержит ли переданный текст оскорбления, нецензурные слова, угрозы или любой другой неподобающий контент. "
        "Верни **строго** JSON-объект с полями 'is_inappropriate' (true/false) и 'reason' (строка с причиной, если is_inappropriate=true, иначе пустая строка). "
        "Пример: {\"is_inappropriate\": true, \"reason\": \"Сообщение содержит нецензурные слова\"} или {\"is_inappropriate\": false, \"reason\": \"\"}. "
        "Не возвращай ничего, кроме JSON-объекта, и убедись, что ключи заключены в двойные кавычки."
    )
    data = {
        "model": "llama3.1-8b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 1000 if not is_moderation else 100,
        "temperature": 0.9 if not is_moderation else 0.3,
        "top_p": 0.9,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(CEREBRAS_API_URL, json=data, headers=headers, timeout=15) as response:
                response.raise_for_status()
                logging.info(f"Остаток лимита API: {response.headers.get('X-RateLimit-Remaining')}")
                result = await response.json()
                response_text = result["choices"][0]["message"]["content"].strip()
                if is_moderation:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logging.error(f"Ошибка парсинга JSON от Cerebras: {response_text}, ошибка: {e}")
                        if not is_retry:
                            retry_prompt = f"Проанализируй этот текст: '{prompt}'. Верни **строго** JSON: {{\"is_inappropriate\": true/false, \"reason\": \"причина или пустая строка\"}}."
                            return await get_cerebras_response(retry_prompt, is_retry=True, is_moderation=True)
                        return {"is_inappropriate": False, "reason": "Ошибка обработки ответа от API"}
                if len(response_text.split()) < 5 and not is_retry:
                    logging.warning(f"Ответ слишком короткий: {response_text}, пробуем доработать")
                    retry_prompt = f"Заверши этот текст в том же весёлом стиле, добавь 2-3 предложения про булочки и чёрный хлеб: {response_text}"
                    return await get_cerebras_response(retry_prompt, is_retry=True)
                return response_text
    except aiohttp.ClientConnectorError as e:
        logging.error(f"Ошибка соединения с Cerebras API: {e}")
        return "Ой, интернет пропал, как мой склад булочек! Проверь соединение и попробуй снова! 😎"
    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            logging.warning("Превышен лимит API. Ждем 10 секунд.")
            await asyncio.sleep(10)
            return "Слишком много запросов! Даже мои булочки так быстро не раскупают! Подожди чуток! 😜"
        logging.error(f"Ошибка API: {e}")
        return "Кажется, Cerebras взял перерыв на булочки. Попробуй снова через минуту! 😎"
    except asyncio.TimeoutError:
        logging.error("Таймаут запроса к Cerebras API")
        return "Cerebras завис, как мой клиент без булочки! Попробуй ещё раз! 😜"

# Проверка на плохие слова
async def check_bad_words(message):
    if len(message.content) < 5:
        return False
    moderation_result = await get_cerebras_response(message.content, is_moderation=True)
    if not isinstance(moderation_result, dict) or 'is_inappropriate' not in moderation_result:
        logging.error(f"Ошибка: Cerebras не вернул ожидаемый JSON для модерации: {moderation_result}")
        return False
    
    if moderation_result['is_inappropriate']:
        await message.delete()
        user_id = message.author.id
        WARNINGS[user_id] = WARNINGS.get(user_id, 0) + 1
        warning_count = WARNINGS[user_id]
        save_warnings()
        
        reason = moderation_result.get('reason', 'Неподобающий контент')
        logging.info(f"Сообщение от {message.author} удалено: {reason}. Предупреждений: {warning_count}")
        
        await message.channel.send(f"{message.author.mention}, эй, не трынди всякую ерунду! Это твоё {warning_count}-е предупреждение. Мой бизнес булочек требует приличного общения! 😎")
        
        if warning_count == MAX_WARNINGS_BEFORE_MUTE:
            try:
                timeout_until = datetime.utcnow() + timedelta(seconds=MUTE_DURATION)
                await message.author.timeout(timeout_until, reason="Слишком много неподобающего контента")
                await message.channel.send(f"{message.author.mention} получил тайм-аут на {MUTE_DURATION // 60} минут за {warning_count} предупреждений. Пора взять паузу и подумать о булочках! 😜")
                logging.info(f"Пользователь {message.author} замьючен на {MUTE_DURATION} секунд")
            except discord.Forbidden:
                await message.channel.send("Не могу замьютить, у меня лапки! Нет прав! 😢")
                logging.error(f"Ошибка: Нет прав для мута {message.author}")
        
        elif warning_count >= MAX_WARNINGS_BEFORE_BAN:
            try:
                await message.author.ban(reason="Слишком много неподобающего контента")
                await message.channel.send(f"{message.author.mention} забанен за {warning_count} предупреждений. Мой склад булочек теперь чище! 😎")
                logging.info(f"Пользователь {message.author} забанен")
                WARNINGS.pop(user_id, None)
                save_warnings()
            except discord.Forbidden:
                await message.channel.send("Не могу забанить, у меня лапки! Нет прав! 😢")
                logging.error(f"Ошибка: Нет прав для бана {message.author}")
        
        return True
    return False

# Команда для создания личного чата
@bot.command()
async def privatechat(ctx):
    guild = ctx.guild
    user = ctx.author
    
    # Проверяем, есть ли уже канал для пользователя
    existing_channel = discord.utils.get(guild.text_channels, name=f"chat-{user.id}")
    if existing_channel:
        await ctx.send(f"Эй, {user.mention}, твой личный чат уже существует: {existing_channel.mention}! Пойдём туда тусить с булочками! 😎")
        return
    
    # Создаём канал
    overwrites = {
        guild.default_role: discord.PermissionOverwrite(read_messages=False),
        user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
        bot.user: discord.PermissionOverwrite(read_messages=True, send_messages=True)
    }
    
    try:
        channel = await guild.create_text_channel(
            name=f"chat-{user.id}",
            overwrites=overwrites,
            topic=f"Личный чат {user.name} с МаксимAI! Обсуждаем булочки и чёрный хлеб! 😜"
        )
        await channel.send(f"{user.mention}, добро пожаловать в твой личный чат! Пиши, что хочешь, а я отвечу с душой и булочками! 😎")
        await ctx.send(f"{user.mention}, твой личный чат создан: {channel.mention}!")
        logging.info(f"Создан личный канал {channel.name} для {user}")
    except discord.Forbidden:
        await ctx.send("Ой, у меня лапки! Нет прав создавать каналы! 😢")
        logging.error(f"Ошибка: Нет прав для создания канала для {user}")
    except Exception as e:
        await ctx.send("Что-то пошло не так, как мой склад булочек! Попробуй снова! 😜")
        logging.error(f"Ошибка при создания канала: {e}")

# HTTP-эндпоинт для команд
async def handle_command(request):
    data = await request.json()
    logging.info(f"Получен запрос на /command: {data}")
    if data.get("api_key") != BOT_API_KEY:
        logging.error(f"Неверный API-ключ: {data.get('api_key')}")
        return web.json_response({"error": "Неверный API-ключ"}, status=401)
    
    guild_id = data.get("guild_id")
    user_id = data.get("user_id")
    action = data.get("action")
    reason = data.get("reason", "Нарушение правил")
    
    guild = bot.get_guild(int(guild_id))
    if not guild:
        logging.error(f"Сервер {guild_id} не найден")
        return web.json_response({"error": "Сервер не найден"}, status=404)
    
    member = guild.get_member(int(user_id))
    if not member:
        logging.error(f"Пользователь {user_id} не найден на сервере {guild_id}")
        return web.json_response({"error": "Пользователь не найден"}, status=404)
    
    try:
        if action == "warn":
            channel = guild.text_channels[0]
            await channel.send(f"{member.mention} получил предупреждение за: {reason}. Не трынди, веди себя как моя лучшая булочка! 😎")
            logging.info(f"Предупреждение выдано: {member} за {reason}")
            return web.json_response({"message": f"Предупреждение выдано {member.name}"})
        
        elif action == "mute":
            timeout_until = datetime.utcnow() + timedelta(seconds=MUTE_DURATION)
            await member.timeout(timeout_until, reason=reason)
            channel = guild.text_channels[0]
            await channel.send(f"{member.mention} получил тайм-аут на {MUTE_DURATION // 60} минут за: {reason}. Пора взять паузу и подумать о булочках! 😜")
            logging.info(f"Пользователь {member} замьючен на {MUTE_DURATION} секунд")
            return web.json_response({"message": f"{member.name} замьючен"})
        
        elif action == "ban":
            await member.ban(reason=reason)
            channel = guild.text_channels[0]
            await channel.send(f"{member.mention} забанен за: {reason}. Мой склад булочек теперь чище! 😎")
            logging.info(f"Пользователь {member} забанен")
            return web.json_response({"message": f"{member.name} забанен"})
        
        else:
            logging.error(f"Неверное действие: {action}")
            return web.json_response({"error": "Неверное действие"}, status=400)
    
    except discord.Forbidden:
        logging.error(f"Нет прав для выполнения действия {action} для {member}")
        return web.json_response({"error": "Нет прав для выполнения действия"}, status=403)
    except Exception as e:
        logging.error(f"Ошибка при выполнении команды: {e}")
        return web.json_response({"error": str(e)}, status=500)

# Запуск HTTP-сервера
async def start_http_server():
    app = web.Application()
    app.add_routes([
        web.post('/command', handle_command),
        web.get('/', lambda _: web.json_response({"message": "Bot is online"}))
    ])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8000)
    await site.start()
    logging.info("HTTP-сервер запущен на порту 8000")

# Бот готов
@bot.event
async def on_ready():
    load_warnings()
    logging.info(f'Бот {bot.user} готов! Подключен к {len(bot.guilds)} серверам')
    try:
        with open('bot_status.txt', 'w') as f:
            f.write(f"{bot.user}|Активен|{len(bot.guilds)}")
    except Exception as e:
        logging.error(f"Ошибка записи статуса: {e}")

# Обработка сообщений
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Проверяем, является ли сообщение в личном канале
    if isinstance(message.channel, discord.TextChannel) and message.channel.name.startswith("chat-"):
        user_id = int(message.channel.name.split("-")[1])
        if message.author.id == user_id:
            response = await get_cerebras_response(message.content)
            await message.channel.send(response)
            return
    
    # Проверяем плохие слова
    if await check_bad_words(message):
        return

    # Обрабатываем команды !chat и !bun
    if message.content.startswith('!chat'):
        prompt = message.content[5:].strip()
        if not prompt:
            await message.channel.send("Эй, братан, напиши что-нибудь после !chat, или мне просто похвалиться новой булочкой? 😎")
            return
        response = await get_cerebras_response(prompt)
        await message.channel.send(response)
    elif message.content.startswith('!bun'):
        prompt = "Расскажи весело и с лёгким юмором про свой бизнес по продаже булочек, упомяни чёрный хлеб, чтобы было забавно и дружелюбно."
        response = await get_cerebras_response(prompt)
        await message.channel.send(response)
    
    await bot.process_commands(message)

# Команды бота
@bot.command()
@commands.has_permissions(manage_messages=True)
async def warn(ctx, member: discord.Member, *, reason="Нарушение правил"):
    await ctx.send(f"{member.mention} получил предупреждение за: {reason}. Не трынди, веди себя как моя лучшая булочка! 😎")
    logging.info(f"Предупреждение выдано: {member} за {reason}")

@bot.command()
@commands.has_permissions(ban_members=True)
async def ban(ctx, member: discord.Member, *, reason="Нарушение правил"):
    await member.ban(reason=reason)
    await ctx.send(f"{member.mention} забанен за: {reason}. Мой склад булочек теперь чище! 😎")
    logging.info(f"Бан выдан: {member} за {reason}")

# Обработка ошибок команд
@warn.error
async def warn_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("Эй, у тебя нет прав! Только боссы могут раздавать предупреждения, а ты пока тренируйся с булочками! 😎")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Кого предупреждать-то? Укажи юзера! Например: !warn @user причина")

@ban.error
async def ban_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("Только главные боссы могут банить! Без прав — иди выбирай булочку! 😜")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Кого банить? Укажи юзера! Например: !ban @user причина")

@privatechat.error
async def privatechat_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("Эй, только боссы могут создавать личные чаты! Иди пока выбирай булочку! 😜")
    else:
        await ctx.send("Что-то сломалось, как мой поднос с булочками! Попробуй снова! 😎")
        logging.error(f"Ошибка команды privatechat: {error}")

# Запуск бота и HTTP-сервера
async def main():
    try:
        await start_http_server()
        await bot.start(DISCORD_BOT_TOKEN)
    except Exception as e:
        logging.error(f"Ошибка запуска: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Бот остановлен пользователем")
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")