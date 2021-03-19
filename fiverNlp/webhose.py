import webhoseio
import datetime

webhoseio.config(token="8018e387-9258-4fd4-9ec5-9f9366a779a8")

today = datetime.datetime.utcnow().date()
epoch = (today - datetime.timedelta(days=7)).strftime('%s')
query_params = {"q": "adial pharmaceuticals language:english ","sort":"crawled","ts": epoch}

output = webhoseio.query("filterWebContent", query_params)
print(333)
for i in output['posts']:
    thread = i.get('thread')
    social = thread.get('social')

webhoseio.config(token="8018e387-9258-4fd4-9ec5-9f9366a779a8")

today = datetime.datetime.utcnow().date()
epoch = (today - datetime.timedelta(days=7)).strftime('%s')
query_params = {"q": "adial pharmaceuticals language:english ","sort":"crawled","ts": epoch}
output = webhoseio.query("filterWebContent", query_params)
print(333)
