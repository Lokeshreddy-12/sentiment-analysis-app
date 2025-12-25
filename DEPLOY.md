# Deployment

This repository includes a Dockerfile so the app can be deployed to any container platform (Render, Fly, Heroku Docker, etc.).

## Deploy to Render (recommended quick start)
1. Sign in to https://dashboard.render.com and create a new **Web Service**.
2. Connect your GitHub repository and choose the `main` branch.
3. Environment: **Docker** (Render will build your `Dockerfile`).
4. Port: `5000`. Render will automatically detect the Dockerfile.
5. Add any required environment variables (none required for the sample model).
6. Deploy. After the first deploy, you’ll have a public URL for the app.

## Deploy to Fly.io (Docker)
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`.
2. `fly launch` within the repo and follow prompts to deploy using the Dockerfile.

## Notes
- For larger models (BERT/TensorFlow), install the `requirements-ml.txt` file on the host or use a larger instance size. See `requirements-ml.txt`.
- If you want automated deploys from GitHub Actions to Render, add a Render service token to repository secrets and I can add a deploy workflow.
