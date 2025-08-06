# Pyodide Express Server

A Node.js Express service that exposes a REST API for executing Python code via [Pyodide](https://pyodide.org/).

## Features
- Execute Python code and return stdout/stderr
- Streaming execution endpoint
- Install additional Python packages at runtime
- Upload files for analysis
- Health and status endpoints
- Modular architecture for adding new routes

## Quick Start
```bash
# Clone and install
git clone https://github.com/brisacoder/pyodide-express-server.git
cd pyodide-express-server
npm ci
cp .env.example .env   # optional
npm start
```

With the server running, try the sample client:
```bash
node examples/basic-client.js
```

## API Endpoints
| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/execute` | Execute Python code and return the output |
| POST | `/api/execute-stream` | Execute code and stream results |
| POST | `/api/install-package` | Install a Python package via `micropip` |
| GET  | `/api/packages` | List installed packages |
| GET  | `/api/status` | Pyodide initialization status |
| GET  | `/api/health` | Python execution health check |
| GET  | `/api/stats` | Server statistics |
| POST | `/api/reset` | Reset the Python environment |
| POST | `/api/upload-csv` | Upload a data file |
| GET  | `/health` | Overall server health |

## Development Workflow
- `npm run dev` – start the server with live reload
- `npm run lint` – lint source files
- `npm run format` – apply Prettier formatting
- Logs live under `logs/` and uploaded files under `uploads/`

See [docs/architecture.md](docs/architecture.md) and [`pyodide_arch.md`](pyodide_arch.md) for a deeper look into the system design.

## Tests
Some tests expect the server to be running locally on port 3000.
```bash
npm test
```

## Contributing
We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License
This project is released under the [MIT License](LICENSE).

## Additional Resources
- [Examples](examples/README.md) – sample clients for interacting with the API.
- [Architecture Overview](docs/architecture.md) – blog-style walkthrough of the design.
