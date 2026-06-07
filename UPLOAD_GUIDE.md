# GitHub Upload Guide — Step by Step

Follow these steps to upload your project to GitHub.

---

## Step 1 — Create GitHub Account / Login
Go to github.com and login.

---

## Step 2 — Create New Repository
1. Click the green **"New"** button (top left)
2. Repository name: `stock-market-crash-warning-system`
3. Description: `KNIME-based ensemble ML system (XGBoost + Keras LSTM) for early detection of stock market crashes using price, volatility, and sentiment data`
4. Set to **Public**
5. ✅ Check "Add a README file" — NO (we have our own)
6. Click **"Create repository"**

---

## Step 3 — Upload Files

### Option A — GitHub Website (Easiest, no Git needed)

1. In your new repo, click **"Add file" → "Upload files"**
2. Upload in this order:
   - First: `README.md` (drag and drop)
   - Commit message: `Add project README`
   - Click "Commit changes"

3. Create folders by clicking **"Add file" → "Create new file"**
   - Type: `workflow/workflow_notes.md`
   - Paste the content from `workflow_notes.md`
   - Commit

4. Repeat for each file:
   - `notebooks/feature_engineering.py`
   - `docs/project_report.md`
   - `data/README.md`
   - `results/model_summary.txt`

5. If you have your actual `.knwf` KNIME workflow file:
   - Upload it to `workflow/` folder
   - This is the most important file for technical reviewers!

---

## Step 4 — Add Topics (Tags)
1. On your repo main page, click the ⚙️ gear icon next to "About"
2. Add topics: `machine-learning`, `xgboost`, `lstm`, `knime`, `financial-data`, `python`, `time-series`
3. These help recruiters find your project

---

## Step 5 — Pin to GitHub Profile
1. Go to your GitHub profile (github.com/your-username)
2. Click "Customize your profile"
3. Pin this repository — it shows on your profile homepage

---

## Step 6 — Update CV Links
Replace "GitHub Link" placeholders in your CV with the actual URL:
`https://github.com/YOUR-USERNAME/stock-market-crash-warning-system`

---

## Your KNWF File
If you have the actual KNIME workflow file (.knwf), upload it to the `workflow/` folder.
This is critical — it's proof the project actually exists and runs.
Export it from KNIME: File → Export KNIME Workflow → Save as .knwf

---

## Final Checklist Before Applying
- [ ] README displays correctly on GitHub (check preview)
- [ ] All links in README work
- [ ] KNWF file uploaded
- [ ] Repository is PUBLIC (not private)
- [ ] Repository pinned on profile
- [ ] LinkedIn updated with GitHub link
- [ ] CV updated with correct GitHub URL
