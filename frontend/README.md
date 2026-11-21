# Frontend (React + Tailwind)

```bash
npm install
npm start        # http://localhost:3000, API calls are proxied to http://localhost:8000
npm run build    # production build in build/, served by the FastAPI backend
```

Set `REACT_APP_API_URL` at build time if the API is hosted on a different domain.
