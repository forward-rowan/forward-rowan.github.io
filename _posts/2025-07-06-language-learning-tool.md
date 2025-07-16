---
layout: post
title: "从零开始：打造个人专属的英日语学习小工具"
date: 2025-07-06 12:00:00 +0800
categories: 项目实践
tags: [JavaScript, Node.js, SQLite, WebDevelopment, PersonalProject, LanguageLearning]
---

## 项目背景

在学习英语和日语的过程中，我发现现有的学习应用要么功能过于复杂，要么不够个性化。于是我决定自己动手，开发一个简单实用的词汇学习工具。

这个项目的核心理念很简单：**遇到不会的词就记录下来，让AI帮忙翻译和造句**。

## 需求分析

### 核心功能
- 📝 快速输入单词、词组或句子
- 🌐 支持英语和日语输入
- 🤖 自动调用AI进行翻译
- 📚 生成实用例句
- 💾 本地存储学习记录
- 🎯 目标语言可选择（中文为主）

### 技术要求
- 轻量级，响应快速
- 跨平台（手机、电脑都能用）
- 离线数据存储
- 低成本或免费运行

## 技术选型

### 为什么选择Web应用？

对于初学者来说，Web应用有几个明显优势：

1. **跨平台兼容**：一套代码，手机电脑都能用
2. **开发门槛低**：HTML、CSS、JavaScript是最容易入门的技术
3. **调试方便**：浏览器开发者工具功能强大
4. **部署简单**：本地运行，无需服务器

### 技术栈选择

```
前端：HTML + CSS + JavaScript (原生)
后端：Node.js + Express
数据库：SQLite
API：免费翻译API + ChatGPT API（可选）
```

**为什么不用框架？**
- 作为初学者，先掌握原生JavaScript更重要
- 项目规模小，框架反而增加复杂度
- 便于理解底层原理

## 系统架构设计

### 整体架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     前端界面     │───→│     后端API     │───→│     数据库      │
│  (用户交互)     │    │  (业务逻辑)     │    │  (数据存储)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │   外部API服务   │
         │              │  (翻译/词典)    │
         └──────────────→└─────────────────┘
```

### 数据流设计

1. **用户输入** → 前端收集数据
2. **数据验证** → 检查输入格式和语言
3. **API调用** → 获取翻译和例句
4. **数据存储** → 保存到本地数据库
5. **结果展示** → 返回给用户界面

## 核心功能实现

### 1. 数据结构设计

```javascript
// 词汇条目的数据结构
const WordEntry = {
    id: Number,           // 唯一标识
    original: String,     // 原始输入
    language: String,     // 语言类型 (en/ja)
    translation: String,  // 翻译结果
    example: String,      // 例句
    timestamp: Date,      // 创建时间
    targetLanguage: String // 目标语言
}
```

### 2. 语言检测

```javascript
function detectLanguage(text) {
    // 检测日语（平假名、片假名、汉字）
    const japaneseRegex = /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]/;
    
    // 检测英语（基本拉丁字母）
    const englishRegex = /^[a-zA-Z\s.,!?'-]+$/;
    
    if (japaneseRegex.test(text)) {
        return 'ja';
    } else if (englishRegex.test(text)) {
        return 'en';
    }
    
    return 'unknown';
}
```

### 3. API集成策略

为了控制成本，我们采用**分层API策略**：

```javascript
// API优先级：免费 → 付费
const apiPriority = [
    'dictionaryapi',    // 免费词典API
    'baiduTranslate',   // 百度翻译（免费额度）
    'chatgpt'           // ChatGPT（付费但效果好）
];
```

### 4. 本地存储方案

```javascript
// 使用SQLite进行本地存储
const sqlite3 = require('sqlite3').verbose();

// 创建数据表
const createTable = `
    CREATE TABLE IF NOT EXISTS words (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original TEXT NOT NULL,
        language TEXT NOT NULL,
        translation TEXT,
        example TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
`;
```

## 开发心得

### 从底层理解计算机原理

在开发过程中，我逐渐理解了一些计算机底层概念：

1. **HTTP协议**：前端和后端通过HTTP请求进行通信
2. **JSON数据格式**：数据在网络中以JSON格式传输
3. **数据库索引**：为了快速查找词汇，需要建立索引
4. **异步编程**：API调用是异步的，需要用Promise处理

### 遇到的技术挑战

1. **跨域问题**：前端调用后端API时的CORS设置
2. **异步处理**：多个API调用的顺序管理
3. **错误处理**：网络异常时的用户体验
4. **数据同步**：多设备间的数据一致性

### 解决方案

```javascript
// 解决跨域问题
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    next();
});

// 优雅的错误处理
async function safeApiCall(apiFunction, fallbackValue) {
    try {
        return await apiFunction();
    } catch (error) {
        console.error('API调用失败:', error);
        return fallbackValue;
    }
}
```

## 项目亮点

### 1. 渐进式开发
- **MVP阶段**：基础的输入和存储功能
- **增强阶段**：添加API集成和智能功能
- **优化阶段**：改进用户体验和性能

### 2. 成本控制
- 优先使用免费API
- 本地存储避免云服务费用
- 按需付费的ChatGPT集成

### 3. 用户体验
- 简洁的界面设计
- 快速的响应速度
- 离线功能支持

## 技术总结

### 学到的知识点

1. **前端开发**：DOM操作、事件处理、异步请求
2. **后端开发**：RESTful API设计、数据库操作
3. **网络编程**：HTTP协议、API调用、错误处理
4. **数据库**：SQLite的使用、SQL语句编写
5. **软件工程**：需求分析、架构设计、测试调试

### 代码组织

```
project/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── backend/
│   ├── server.js
│   ├── database.js
│   └── api.js
├── database/
│   └── words.db
└── README.md
```

## 未来规划

### 短期目标
- [ ] 完善词汇管理功能
- [ ] 添加学习统计
- [ ] 优化移动端体验

### 长期目标
- [ ] 智能复习算法
- [ ] 语音识别输入
- [ ] 云端同步功能

## 给其他初学者的建议

1. **从简单开始**：先实现核心功能，再添加高级特性
2. **多动手实践**：理论知识要通过实际项目来巩固
3. **善用免费资源**：很多优秀的开源工具和免费API可以使用
4. **记录开发过程**：写技术博客能帮助理清思路
5. **不要害怕出错**：每个错误都是学习的机会

## 结语

这个项目让我深刻体会到了软件开发的乐趣。从最初的想法到最终的实现，每一步都充满了挑战和收获。虽然作为初学者，代码可能不够优雅，但是解决实际问题的成就感是无可替代的。

希望这篇文章能给其他想要开发个人项目的同学一些启发。记住：**最好的学习方式就是动手实践**！

---

**技术标签：** #JavaScript #Node.js #SQLite #WebDevelopment #PersonalProject #LanguageLearning