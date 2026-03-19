FROM node:20-alpine
WORKDIR /app
COPY package.json .
RUN npm config set strict-ssl false
RUN npm install
COPY . .
CMD ["npm", "run", "start"]