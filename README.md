
# AI Forecasting Web App (Node.js + React)

A production-ready starter to ingest client-uploaded data (CSV/XLSX), parse to a standard schema,
run **time-series forecasting** (ARIMA), and visualize results on a **React** dashboard.

## Stack
- **Backend** (server): Node.js + TypeScript + Express, Multer uploads, CSV/XLSX parsing, ARIMA, Zod validation, CORS
- **Frontend** (web): Vite + React + TypeScript + Tailwind + Recharts + React Hook Form + Zod
- **Infra**: Docker Compose for local dev; GitHub Actions CI

## Quick Start
```bash
# 1) Clone and bootstrap
npm run bootstrap

# 2) Start all (server + web)
npm run dev

# Server: http://localhost:4000/api/health
# Web:    http://localhost:5173
```

### Upload and forecast (via UI)
- Open the web app → Upload your CSV/XLSX
- Choose **Date column**, **Value column**, **Frequency** (D/W/M) and **Horizon**
- Submit to get forecast + chart + downloadable CSV

### API (cURL)
```bash
curl -X POST "http://localhost:4000/api/forecast"   -F "file=@samples/monthly_sales.csv"   -F "dateCol=Date"   -F "valueCol=Sales"   -F "frequency=M"   -F "horizon=6"
```

## Repo Layout
```
server/       Node + TS backend (Express)
web/          React + Vite frontend
infra/        docker-compose.yml
.github/      CI workflows
samples/      Example datasets
```

## Next Steps
- Add auth and per-client storage
- Persist uploaded files to S3 and forecasts to DB
- Support Prophet (Python microservice) option for advanced seasonality
- Add model registry & experiment tracking
