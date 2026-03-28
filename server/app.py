# server/app.py
# Required by OpenEnv validator — must have a callable main() function
# and if __name__ == '__main__' block.

import uvicorn
from app import app

def main():
	"""Entry point for the OpenEnv server."""
	uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
	main()
