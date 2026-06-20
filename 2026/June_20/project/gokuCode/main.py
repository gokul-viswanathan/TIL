import requests
from pathlib import Path
import os


def loadenv():

    for line in Path(".env.local").read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip()


def request_ai(user_input):
    url = os.getenv("gemini-endpoint", "")
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": os.getenv("google-api-key"),
    }
    data = {"contents": user_input}
    try:
        resp = requests.post(url, headers=headers, json=data)
        print(resp)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def main():
    chat_history = []
    while True:
        try:
            user_input = input("Users question :")
        except EOFError, KeyboardInterrupt:
            print()
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break

        chat_history.append({"role": "user", "parts": [{"text": user_input}]})
        response = request_ai(chat_history)
        text_output = response["candidates"][0]["content"]["parts"][0]["text"]
        chat_history.append(
            {
                "role": "model",
                "parts": [{"text": text_output}],
            }
        )
        print(
            "ai response",
            text_output,
            end="\n",
        )


if __name__ == "__main__":
    loadenv()
    main()
