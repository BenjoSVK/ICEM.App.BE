# ICEM Application (Backend)

## How to access endpoints

Example:
`http://localhost:7030/ikem_api/test`

Links for endpoints documentation (server needst to be started first): `http://localhost:7030/docs/`


## How to start application?

### Local

`uvicorn src.api.main:app --reload`

***Info***: needs to be located in a main folder (cd ICEM.App.BE)

## Docker

Initial create of network:

```bash
docker network create vgg_histo_network
docker volume create db_data
```

### Development
Uses `.env`. From repo root:

```bash
docker compose -f docker-compose.dev.yml up
```

Rebuild:

```bash
docker compose -f docker-compose.dev.yml up --build
```

### Production

```bash
docker compose -f docker-compose.prod.yaml up -d
```

Rebuild:

```bash
docker compose -f docker-compose.prod.yaml up -d --build
```

# Init testing database
Passwords are stored as **bcrypt hashes**.

`psql -U postgres -d mydatabase`:

```sql
CREATE TABLE IF NOT EXISTS "users" (
    username VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);
```

Generate a bcrypt hash for your password (e.g. `admin`), then insert it:

```bash
python3 -c "import bcrypt; print(bcrypt.hashpw(b'admin', bcrypt.gensalt()).decode())"
```

Copy the output and use it in the INSERT (replace `PASTE_HASH_HERE` with the actual hash):

```sql
INSERT INTO "users" (username, password) VALUES ('admin', 'PASTE_HASH_HERE');
SELECT * FROM "users";
```