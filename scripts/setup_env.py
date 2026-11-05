from pathlib import Path
import secrets

p = Path(__file__).resolve().parents[1] / ".env"
if p.exists():
    print(".env already exists; unchanged.")
else:
    p.write_text(
        "MONGO_ADMIN_PASSWORD="
        + secrets.token_hex(24)
        + "\nMONGO_APP_PASSWORD="
        + secrets.token_hex(24)
        + "\nAPI_KEY="
        + secrets.token_hex(24)
        + "\nMONGO_HOST=localhost\n",
        encoding="utf-8",
    )
    print("Created .env. Copy API_KEY into the local UI. Never commit this file.")
