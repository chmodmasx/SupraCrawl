FROM node:22-bookworm-slim

ENV NODE_ENV=production \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev && npx playwright install --with-deps chromium
COPY server.mjs ./

RUN useradd --create-home --uid 10001 supracrawl \
    && mkdir -p /ms-playwright \
    && chown -R supracrawl:supracrawl /app /ms-playwright
USER supracrawl

EXPOSE 3000
CMD ["node", "server.mjs"]
