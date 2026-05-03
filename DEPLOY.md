# AnomaQ — Deploy to Render (Free)

## Step 1 — Push to GitHub

If you don't have a GitHub account: https://github.com/signup

```bash
# In the anomaq_app folder:
git init
git add .
git commit -m "Initial AnomaQ commit"
```

Then create a new repo at https://github.com/new (name it `anomaq`, keep it public or private — both work on Render free tier).

```bash
git remote add origin https://github.com/YOUR_USERNAME/anomaq.git
git branch -M main
git push -u origin main
```

---

## Step 2 — Deploy on Render

1. Go to **https://render.com** and sign up (free, use GitHub login)
2. Click **"New +"** → **"Web Service"**
3. Click **"Connect a repository"** → select your `anomaq` repo
4. Render will auto-detect the `render.yaml` — just confirm these settings:
   - **Name**: anomaq (or anything you like)
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Click **"Create Web Service"**

---

## Step 3 — Wait for build (~5 min first time)

Render will install all dependencies including Qiskit. You'll see a live build log.

When you see:
```
==> Your service is live 🎉
```

Your app is live at: `https://anomaq.onrender.com` (or similar)

---

## Notes

- **Cold starts**: Free tier spins down after 15min of inactivity. First request after that takes ~30s to wake up. Subsequent requests are instant.
- **Re-deploys**: Every `git push` to `main` triggers an automatic redeploy.
- **Logs**: Check Render dashboard → your service → "Logs" tab for any errors.

---

## File structure (what you should push)

```
anomaq_app/
├── main.py           ← FastAPI backend
├── index.html        ← Frontend UI
├── requirements.txt  ← Python dependencies
└── render.yaml       ← Render config
```
