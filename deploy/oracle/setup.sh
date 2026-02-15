#!/bin/bash
# ATLAS Oracle Cloud Setup Script
# Run on a fresh Oracle Cloud ARM A1 instance (Ubuntu 22.04)
#
# Usage: bash setup.sh [DOMAIN]
# Example: bash setup.sh atlas.example.com

set -euo pipefail

DOMAIN="${1:-localhost}"
APP_DIR="/opt/atlas"

echo "=== ATLAS Oracle Cloud Setup ==="
echo "Domain: $DOMAIN"
echo ""

# ── 1. System Updates ─────────────────────────────────────────────
echo "[1/7] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# ── 2. Install Docker ─────────────────────────────────────────────
echo "[2/7] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
fi

# Install Docker Compose plugin
if ! docker compose version &> /dev/null; then
    sudo apt-get install -y -qq docker-compose-plugin
fi

# ── 3. Create app directory ──────────────────────────────────────
echo "[3/7] Setting up application directory..."
sudo mkdir -p "$APP_DIR"
sudo chown "$USER:$USER" "$APP_DIR"

# ── 4. Clone / copy application ──────────────────────────────────
echo "[4/7] Copying application files..."
if [ -d ".git" ]; then
    cp -r . "$APP_DIR/"
else
    echo "  Run this script from the project root directory."
    exit 1
fi

cd "$APP_DIR"

# ── 5. Create .env file ─────────────────────────────────────────
echo "[5/7] Creating environment configuration..."
if [ ! -f "$APP_DIR/.env" ]; then
    _PG_PASS=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)
    _JWT_SEC=$(openssl rand -base64 32 | tr -d '/+=' | head -c 48)
    _REDIS_PASS=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)
    cat > "$APP_DIR/.env" << ENVEOF
# Database (auto-generated — keep secret)
POSTGRES_PASSWORD=${_PG_PASS}

# API Authentication
ATLAS_API_KEY=
ATLAS_JWT_SECRET=${_JWT_SEC}

# Redis Authentication
REDIS_PASSWORD=${_REDIS_PASS}

# LLM API Keys (add yours)
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Domain (for SSL)
DOMAIN=localhost
ENVEOF
    echo "  Created .env file — edit it with your API keys!"
    echo "  nano $APP_DIR/.env"
fi

# ── 6. Generate self-signed SSL (replaced by certbot later) ─────
echo "[6/7] Setting up SSL certificates..."
sudo mkdir -p /etc/letsencrypt/live/atlas
if [ ! -f /etc/letsencrypt/live/atlas/fullchain.pem ]; then
    sudo openssl req -x509 -nodes -days 365 \
        -newkey rsa:2048 \
        -keyout /etc/letsencrypt/live/atlas/privkey.pem \
        -out /etc/letsencrypt/live/atlas/fullchain.pem \
        -subj "/CN=$DOMAIN" 2>/dev/null
    echo "  Self-signed cert created. Run certbot for production SSL."
fi

# ── 7. Start services ───────────────────────────────────────────
echo "[7/7] Starting ATLAS stack..."
cd "$APP_DIR/deploy/oracle"
docker compose -f docker-compose.prod.yml up -d

echo ""
echo "=== ATLAS Setup Complete ==="
echo ""
echo "Services:"
echo "  Backend:  http://localhost:8000"
echo "  Postgres: localhost:5432"
echo "  Redis:    localhost:6379"
echo "  Nginx:    https://$DOMAIN"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys: nano $APP_DIR/.env"
echo "  2. For production SSL: sudo certbot certonly --webroot -w /var/www/certbot -d $DOMAIN"
echo "  3. View logs: docker compose -f docker-compose.prod.yml logs -f"
echo "  4. Stop: docker compose -f docker-compose.prod.yml down"
