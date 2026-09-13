# Astro splat viewer → static build, served by nginx.
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
# .splat / .ply have no registered MIME type → nginx serves them as the default
# application/octet-stream, which is exactly what the loader wants. No extra config.
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
