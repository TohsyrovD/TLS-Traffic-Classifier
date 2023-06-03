import json
import requests
import pandas as pd

# Обозначим заголовки для POST-запроса к серверу и для получения от него ответа
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

# Импорт тестового набора данных
file_x = input('Введите нужный файл для его классификации:')
reduced_x = pd.read_csv(file_x, encoding="utf-8-sig")

# Берем первые 10 строк (можно брать все)
df = reduced_x.head()

# Преобразуем датафрейм pandas в json
data = df.to_json(orient='records')

# POST <url>/predict
resp = requests.post("http://192.168.1.130:8000/predict", \
                    data = json.dumps(data),\
                    headers= header)

# Выводим ответ сервера с предсказаниями
print(resp.json())