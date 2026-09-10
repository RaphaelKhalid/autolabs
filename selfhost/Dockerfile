FROM node:24-bookworm-slim AS dependencies
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM dependencies AS build
COPY . .
ENV AUTOLABS_SELF_HOSTED=1
ENV AUTOLABS_STANDALONE=1
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:24-bookworm-slim AS web
WORKDIR /app
ENV NODE_ENV=production
ENV AUTOLABS_SELF_HOSTED=1
ENV NEXT_TELEMETRY_DISABLED=1
ENV HOSTNAME=0.0.0.0
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]

FROM node:24-bookworm-slim AS runner
WORKDIR /app
COPY --chown=node:node selfhost ./selfhost
RUN mkdir /data && chown node:node /data
ENV AUTOLABS_RUNNER_HOST=0.0.0.0
ENV AUTOLABS_DATA_DIR=/data
USER node
CMD ["node", "selfhost/server.ts"]
