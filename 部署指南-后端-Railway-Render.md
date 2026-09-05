# 后端部署指南（Railway / Render）

> 为什么后端要迁走：Netlify 前端页面是 HTTPS，浏览器**只会**调用「正规 HTTPS」的后端地址。
> 你原来的阿里云服务器 `https://8.222.129.233` 用的是**自签名证书**，浏览器一律拒绝
> （`ERR_CERT_AUTHORITY_INVALID`），而且它的 nginx 把 `/api/` 转发到了 8090 的另一个项目，
> 不是 ACA 后端（8000）。所以前端调到不了后端——不是后端没运行，是**证书不受信任 + 路由不对**。
>
> Railway / Render 免费档会给你的后端一个 `https://...` 正规域名（自动 TLS 证书），一举解决证书问题。

## 这个后端是什么
- FastAPI 应用（`main.py`），提供 OA / ACA / MaxDiff 三套接口。
- **无状态、无数据库**：ACA 会话存在内存里，重启即清。适合 PaaS 免费档。
- 依赖：`numpy cvxpy scipy fastapi uvicorn`（见 `requirements.txt`）。
- 端口：读取环境变量 `$PORT`（Railway/Render 自动注入），本地默认 8000。

## 二选一：Railway 或 Render

### 方式 A —— Railway（推荐）
1. 打开 https://railway.com ，用 GitHub 登录。
2. 右上 **New Project → Deploy from GitHub repo**（或先 `railway login` + `railway link` + `railway up`，从本文件夹直接上传）。
   - 若用 GitHub：把本 `backend` 文件夹推到一个仓库根目录（即仓库根含 `main.py`），Railway 选那个仓库。
3. Railway 检测到 `Dockerfile` → 用 Docker 构建（已改好，读 `$PORT`）。也可以直接让它用默认 Python 构建 + Procfile 的启动命令。
4. 部署完成后得到一个域名，形如：`https://aca-backend-production.up.railway.app`。
5. 验证：浏览器打开 `https://<你的域名>/docs`，能看到 FastAPI Swagger 页面即可。
   - 根路径 `/` 返回 `{"status":"ok","services":["oa-generator","aca-service","maxdiff"]}` 也说明正常。

### 方式 B —— Render
1. 打开 https://render.com ，注册（可 GitHub 登录）。
2. **New + → Web Service**，连接仓库（把本 `backend` 文件夹作为仓库根）。
3. 环境：
   - **Runtime / Build**：检测到 `Dockerfile` 时选 **Docker**；若不用 Docker，选 Python 3，
     Build Command 留空、Start Command 填 `uvicorn main:app --host 0.0.0.0 --port $PORT`。
4. 免费实例会休眠（15 分钟无人访问后冷启动变慢），可接受。
5. 部署完得到 `https://<名字>.onrender.com`，同样打开 `/docs` 验证。

## 部署后要做的事
1. **记下后端 HTTPS 地址**（形如 `https://xxx.up.railway.app` 或 `https://xxx.onrender.com`）。
2. 打开 `frontend/部署指南-前端-Netlify.md`，把该地址填进前端 `.env.production`，重建并上传 Netlify。

## 常用检查
- 后端单测（本机即可跑）：`python test_aca_engine.py`（7 个用例，覆盖基准编码/顺序约束/首题平衡/题数=p 等）。
- 日志：Railway / Render 后台的 Logs 面板。
- 改代码后：git push → 平台自动重新部署（或 `railway up`）。
