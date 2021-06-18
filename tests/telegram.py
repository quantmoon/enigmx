import requests

def send_message(message): ###Mensaje de Telegram --- No importante
    bot_token = "1447482605:AAEKvAzUAysfgBbf20YBHXbY1fsu1zYE5Hc"
    chat_ID = "-1001431989466"
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_ID + '&text=' + message
    response = requests.get(send_text)
    return response
