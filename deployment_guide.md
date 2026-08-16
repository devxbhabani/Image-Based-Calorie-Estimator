# 🚀 Deployment Guide: Frontend (Vercel) & Backend (Render)

This guide walks you through deploying your full-stack Image-Based Food Calorie Estimator project.

---

## 🎨 1. Frontend Deployment (Vercel)

Vercel is optimized for frontend hosting. Follow these steps to deploy:

### **Step A: Push Changes to GitHub**
Ensure your local project changes (especially the environment variable updates in React) are committed and pushed:
```bash
git add -A
git commit -m "Configure production API URLs and dependencies"
git push
```

### **Step B: Create a Vercel Project**
1. Log in to [Vercel](https://vercel.com/) (sign in with your GitHub account).
2. Click **Add New** > **Project**.
3. Import your GitHub repository: `Image-Based-Calorie-Estimator`.

### **Step C: Configure Build Settings**
In the project configuration page, set the following:
*   **Framework Preset:** Vite
*   **Root Directory:** `frontend/Food-Calorie-Estimator` (Click Edit and select this folder)
*   **Build Command:** `npm run build`
*   **Output Directory:** `dist`

### **Step D: Configure Environment Variables**
Expand the **Environment Variables** section and add the URL of your deployed backend:
*   **Key:** `VITE_API_URL`
*   **Value:** `https://your-backend-service.onrender.com` *(Replace this with your actual Render URL after deploying the backend in Section 2)*

Click **Deploy**.

---

## 🐍 2. Backend Deployment (Render)

Render is used to host your Python Flask API server.

### **Step A: Configure a Render Web Service**
1. Log in to [Render](https://render.com/) (sign in with your GitHub account).
2. Click **New** > **Web Service**.
3. Link your GitHub repository: `Image-Based-Calorie-Estimator`.

### **Step B: Configure Build & Start Commands**
On the settings page, fill in the following parameters:
*   **Name:** `image-based-calorie-estimator`
*   **Region:** Select the region closest to you
*   **Branch:** `main`
*   **Root Directory:** `backend` (Render will build inside the `backend` folder)
*   **Runtime:** `Python 3`
*   **Build Command:** `pip install -r requirements.txt`
*   **Start Command:** `gunicorn app:app` (Gunicorn is the WSGI server Render uses to run Flask production apps)

### **Step C: Choose Instance Type & Warning on Free Tier Memory**
*   **Warning:** Render's free tier has a **512 MB RAM limit**. Loading PyTorch models (like ResNet50 and MiDaS) inside memory can exceed this limit and trigger an Out of Memory (OOM) crash.
*   **Workaround:** We have built a fallback mechanism in `app.py`. If the model weights fail to load due to RAM limits, the server will continue running on a fallback volume-density estimation model to avoid crashing.
*   **Production recommendation:** For optimal performance, choose the **Starter Plan** (2 GB RAM for $7/month) to run the machine learning models at full accuracy.

Click **Create Web Service**.

---

## 🔗 3. Connect Frontend to Backend

1. Once Render finishes building, copy the live URL of your backend (it looks like `https://image-based-calorie-estimator.onrender.com`).
2. Go back to your **Vercel Dashboard** > **Project Settings** > **Environment Variables**.
3. Update the `VITE_API_URL` variable with your copied Render URL.
4. Redeploy the frontend in Vercel to apply the updated environment variable.
