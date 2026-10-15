# Frontend

React UI for the droplet backend, built with [Vite](https://vite.dev/). Setup and the two ways to run it (served by the backend, or the hot-reload dev server) are in the [root README](../README.md).

## Scripts

### `npm start`

Runs the dev server on [http://localhost:3000](http://localhost:3000), with hot reload. It needs `frontend/.env` to point at the backend (see `.env.example`):

```
REACT_APP_BACKEND_API_URL=http://localhost:8050/api
```

Variables prefixed `REACT_APP_` (or `VITE_`) are exposed to the app as `import.meta.env.<NAME>`.

### `npm run build`

Writes the production bundle to `build/`, which the backend serves at `/` (and the Docker image copies in). Without a `.env`, the app calls the API at the relative `/api`.

### `npm run preview`

Serves the contents of `build/` locally, to check a production build without the backend.
