FROM node:20-alpine

WORKDIR /app

# Copy package files first (for better caching)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy app source code *after* installing deps
COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
