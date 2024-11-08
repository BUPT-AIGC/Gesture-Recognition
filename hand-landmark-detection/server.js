const express = require('express');
const path = require('path');

const app = express();
const port = 7872; // 你可以根据需要修改端口

// 设置静态文件服务，指向当前目录
app.use(express.static(path.join(__dirname)));

// 启动服务器
app.listen(port, () => {
  console.log(`服务器已启动，访问地址：http://localhost:${port}`);
});