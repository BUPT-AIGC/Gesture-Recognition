// app.js
const express = require('express');
const path = require('path');

// 创建 express 应用
const app = express();

// 设置静态文件目录
app.use(express.static(path.join(__dirname, 'public')));

// 处理根路径请求，返回 index.html
app.get('/', function (req, res) {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 创建 HTTP 服务器
const HTTP_PORT = 8000;
app.listen(HTTP_PORT, function () {
  console.log('HTTP Server is running on: http://localhost:%d', HTTP_PORT);
});