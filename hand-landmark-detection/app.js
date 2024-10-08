// 导入必需的模块
const fs = require('fs');
const https = require('https');
const express = require('express');
const path = require('path');

// 读取 SSL 证书和密钥
const privateKey = fs.readFileSync('./172.20.10.2_key.txt', 'utf8');
const certificate = fs.readFileSync('./172.20.10.2_ssl.crt', 'utf8');

// 设置证书凭据
const credentials = { key: privateKey, cert: certificate };

// 创建 express 应用
const app = express();

// 设置静态文件目录
app.use(express.static(path.join(__dirname)));

// 处理根路径请求，返回 index.html
app.get('/', function(req, res) {
  res.sendFile(path.join(__dirname, 'index.html'));
  // res.send("hello");
});

// 创建 HTTPS 服务器
const httpsServer = https.createServer(credentials, app);

// 监听 HTTPS 服务器端口
const SSLPORT = 8001;
httpsServer.listen(SSLPORT, function() {
  console.log('HTTPS Server is running on: https://localhost:%d', SSLPORT);
});