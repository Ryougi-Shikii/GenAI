import os, base64, requests
from dotenv import load_dotenv
load_dotenv()

acc = os.environ["CF_ACCOUNT_ID"].strip()
tok = os.environ["CF_API_TOKEN"].strip()
print(f"token len={len(tok)} prefix={tok[:6]!r}")

r = requests.get("https://api.cloudflare.com/client/v4/accounts",
                 headers={"Authorization": f"Bearer {tok}"})
print(r.json())   # list of accounts this token can see
