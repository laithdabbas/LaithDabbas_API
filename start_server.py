import os
import threading
import time

import uvicorn

from app.main import app
import app.demo as demo_mod


def main() -> int:
    # Keep the default model small unless the user overrides it.
    os.environ.setdefault("GENERAL_LLM_MODEL", "tinyllama")

    print("Starting FastAPI on http://localhost:8000 ...")
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    api_thread = threading.Thread(target=server.run)
    api_thread.start()

    # Give Uvicorn a moment to bind before starting Gradio.
    time.sleep(0.8)

    print("Starting Gradio on http://localhost:7860 ...")
    try:
        demo_mod.demo.launch(server_name="0.0.0.0", server_port=7860)
    finally:
        server.should_exit = True
        api_thread.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
