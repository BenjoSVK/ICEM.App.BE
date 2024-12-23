#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE TABLE users (username VARCHAR(50), password VARCHAR(50));
    INSERT INTO users (username, password) VALUES ('admin', 'admin');
EOSQL