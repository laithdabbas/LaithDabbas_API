from app.main import app
import uvicorn

# test push: add a harmless comment to validate write access
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
