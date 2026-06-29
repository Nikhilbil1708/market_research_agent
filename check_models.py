import anthropic, os
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

models = client.models.list()
for m in models.data:
    print(m.id)
