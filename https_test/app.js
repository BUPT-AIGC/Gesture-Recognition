//导入模块
const fs = require('fs')// 文件输入输出，用来导入证书        
const https = require('https')// https服务器  
const express = require('express')// express模块导入
 
//读取证书
const privateKey = fs.readFileSync('C:\\Users\\Tony\\Desktop\\HongMeng\\https_test\\172.20.10.2_key.txt', 'utf8')
const certificate = fs.readFileSync('C:\\Users\\Tony\\Desktop\\HongMeng\\https_test\\172.20.10.2_ssl.crt', 'utf8')
 
// 创建 express 应用
const app = express()
// 监听 / 路径的 get 请求
app.get('/', function(req, res) {
  res.send('Hello Word!')
})
 
 
const credentials = { key: privateKey, cert: certificate }
const httpsServer = https.createServer(credentials, app)
const SSLPORT = 443
httpsServer.listen(SSLPORT, function() {
  console.log('HTTPS Server is running on: https://localhost:%s', SSLPORT)
})