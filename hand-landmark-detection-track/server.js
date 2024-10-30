const express = require('express');
const path = require('path');

const app = express();
const PORT = 7872;

// 设置静态文件目录为当前目录，即 hand-landmark-detection-track 文件夹
app.use(express.static(path.join(__dirname)));

// 当用户访问根路径时，返回 index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});