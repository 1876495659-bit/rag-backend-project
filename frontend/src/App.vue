/**
 * RAG 知识库问答系统 - 根组件
 *
 * 功能：
 * 1. 管理对话消息列表
 * 2. 处理用户输入和发送请求
 * 3. 处理流式响应（打字机效果）
 * 4. 自动滚动到最新消息
 */

<template>
  <div class="chat-app">
    <!-- 顶部导航栏 -->
    <header class="chat-header">
      <div class="header-left">
        <el-icon class="logo-icon"><Promotion /></el-icon>
        <h1>知识库问答系统</h1>
      </div>
      <div class="header-right">
        <el-tag :type="isConnected ? 'success' : 'danger'" size="small">
          {{ isConnected ? '已连接' : '未连接' }}
        </el-tag>
        <el-button type="danger" size="small" @click="clearChat" :icon="Delete">
          清空对话
        </el-button>
      </div>
    </header>

    <!-- 聊天消息区域 -->
    <main class="chat-main" ref="messageContainer">
      <div v-if="messages.length === 0" class="empty-state">
        <el-icon class="empty-icon"><ChatDotRound /></el-icon>
        <p>欢迎使用知识库问答系统</p>
        <p class="empty-hint">请输入您的问题，或上传文档开始使用</p>
        <el-button type="primary" @click="showDemoQuestion">
          查看示例问题
        </el-button>
      </div>

      <div v-for="(msg, index) in messages" :key="index" class="message-wrapper">
        <!-- 用户消息 -->
        <div v-if="msg.role === 'user'" class="message user-message">
          <div class="avatar user-avatar">
            <el-icon><User /></el-icon>
          </div>
          <div class="bubble user-bubble">
            <div class="message-text">{{ msg.content }}</div>
            <div class="message-time">{{ msg.time }}</div>
          </div>
        </div>

        <!-- AI 助手消息 -->
        <div v-else class="message ai-message">
          <div class="avatar ai-avatar">
            <el-icon><Promotion /></el-icon>
          </div>
          <div class="bubble ai-bubble">
            <div class="message-text">{{ msg.content }}</div>
            <!-- 引用来源展示 -->
            <div v-if="msg.references && msg.references.length > 0" class="references">
              <div class="ref-title">
                <el-icon><Link /></el-icon> 引用来源：
              </div>
              <div v-for="(ref, refIdx) in msg.references" :key="refIdx" class="ref-item">
                <el-tag size="small" type="info">{{ ref.file_name }}</el-tag>
                <span class="ref-score">相关度: {{ ref.score }}</span>
              </div>
            </div>
            <!-- 响应时间 -->
            <div v-if="msg.responseTime" class="response-time">
              耗时: {{ msg.responseTime }}秒
            </div>
            <div class="message-time">{{ msg.time }}</div>
          </div>
        </div>
      </div>

      <!-- 加载中指示器 -->
      <div v-if="isGenerating" class="message ai-message loading-indicator">
        <div class="avatar ai-avatar">
          <el-icon><Promotion /></el-icon>
        </div>
        <div class="bubble ai-bubble loading-bubble">
          <div class="typing-dots">
            <span></span><span></span><span></span>
          </div>
          <span class="loading-text">正在思考...</span>
        </div>
      </div>
    </main>

    <!-- 底部输入区域 -->
    <footer class="chat-footer">
      <!-- 上传文档按钮 -->
      <div class="upload-area">
        <el-upload
          ref="uploadRef"
          :action="`http://localhost:8000/upload_batch`"
          :headers="uploadHeaders"
          :before-upload="beforeUpload"
          :on-success="handleUploadSuccess"
          :on-error="handleUploadError"
          :show-file-list="false"
          accept=".pdf,.docx"
          multiple
        >
          <el-tooltip content="上传 PDF 或 Word 文档到知识库" placement="top">
            <el-button :icon="Upload" circle title="上传文档" />
          </el-tooltip>
        </el-upload>
      </div>

      <!-- 输入框和发送按钮 -->
      <div class="input-area">
        <el-input
          v-model="inputQuery"
          type="textarea"
          :rows="2"
          placeholder="输入您的问题... 按 Shift+Enter 换行，Enter 发送"
          :disabled="isGenerating"
          resize="none"
          @keydown="handleKeydown"
        />
        <div class="input-actions">
          <div class="left-info">
            <span v-if="inputQuery.trim()" class="char-count">{{ inputQuery.trim().length }} 字</span>
          </div>
          <el-button
            type="primary"
            :icon="Promotion"
            :loading="isGenerating"
            :disabled="!inputQuery.trim()"
            @click="sendMessage"
            round
          >
            发送
          </el-button>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Promotion, User, ChatDotRound, Delete,
  Upload, Link
} from '@element-plus/icons-vue'

// ==========================================
// 状态管理
// ==========================================

/** 连接状态 */
const isConnected = ref(false)

/** 消息列表 */
const messages = ref([])

/** 输入框内容 */
const inputQuery = ref('')

/** 是否正在生成回答 */
const isGenerating = ref(false)

/** 消息容器引用（用于自动滚动） */
const messageContainer = ref(null)

/** 上传组件引用 */
const uploadRef = ref(null)

/** 上传请求头（CORS 跨域需要） */
const uploadHeaders = {
  'Accept': 'application/json'
}

/** 示例问题列表 */
const demoQuestions = [
  '请总结知识库中的主要内容',
  '知识库中有哪些技术文档？',
  '请解释相关的概念',
  '知识库中包含哪些案例？'
]

/** 当前选中的示例问题索引 */
let currentDemoIndex = 0

// ==========================================
// 自动滚动
// ==========================================

/**
 * 滚动到最新消息
 * 在消息变化时自动触发
 */
const scrollToBottom = async () => {
  await nextTick()
  if (messageContainer.value) {
    messageContainer.value.scrollTop = messageContainer.value.scrollHeight
  }
}

// 监听消息变化，自动滚动到底部
watch(messages, scrollToBottom, { deep: true })

// ==========================================
// 示例问题功能
// ==========================================

/**
 * 显示下一个示例问题
 */
const showDemoQuestion = () => {
  inputQuery.value = demoQuestions[currentDemoIndex]
  currentDemoIndex = (currentDemoIndex + 1) % demoQuestions.length
}

// ==========================================
// 消息发送
// ==========================================

/**
 * 发送用户消息
 */
const sendMessage = async () => {
  const query = inputQuery.value.trim()
  if (!query || isGenerating.value) return

  // 添加用户消息到消息列表
  messages.value.push({
    role: 'user',
    content: query,
    time: getCurrentTime()
  })

  // 清空输入框
  inputQuery.value = ''
  isGenerating.value = true

  try {
    // 调用流式 API
    await fetchStreamAnswer(query)
  } catch (error) {
    ElMessage.error(`发送失败: ${error.message}`)
    // 移除错误的 AI 消息
    messages.value.pop()
  } finally {
    isGenerating.value = false
  }
}

/**
 * 获取流式回答（打字机效果）
 *
 * 使用 Fetch API 接收 Server-Sent Events (SSE) 流式数据，
 * 实时更新 AI 回复内容，实现 ChatGPT 式的打字机效果。
 */
const fetchStreamAnswer = async (question) => {
  // 创建占位的 AI 消息
  const aiMessageIndex = messages.value.length
  messages.value.push({
    role: 'assistant',
    content: '',
    references: [],
    responseTime: 0,
    time: getCurrentTime()
  })

  try {
    // 构建 API 请求
    const formData = new FormData()
    formData.append('question', question)
    formData.append('top_k', '3')
    formData.append('use_rerank', 'true')

    const response = await fetch('http://localhost:8000/query/stream', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`HTTP 错误: ${response.status}`)
    }

    // 读取流式响应
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let fullAnswer = ''
    let references = []
    let responseTime = 0

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      // 解码字节流为文本
      const text = decoder.decode(value, { stream: true })

      // 解析 SSE 格式的数据
      const lines = text.split('\n')
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.substring(6))

            if (data.type === 'answer') {
              // 累积回答内容
              fullAnswer += data.content
              messages.value[aiMessageIndex].content = fullAnswer
              await nextTick()
            } else if (data.type === 'done') {
              // 生成完成
            } else if (data.type === 'error') {
              // 处理错误
              fullAnswer += `\n（错误: ${data.content}）`
              messages.value[aiMessageIndex].content = fullAnswer
            }
          } catch (e) {
            // 忽略解析错误
          }
        }
      }
    }

    // 调用普通 API 获取引用来源（因为流式 API 不返回引用）
    try {
      const searchResponse = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          question,
          top_k: '3',
          use_rerank: 'true'
        })
      })

      if (searchResponse.ok) {
        const result = await searchResponse.json()
        references = result.references || []
        responseTime = result.response_time || 0
      }
    } catch (e) {
      // 引用获取失败不影响回答
    }

    // 更新 AI 消息（添加引用和耗时）
    messages.value[aiMessageIndex].references = references
    messages.value[aiMessageIndex].responseTime = responseTime

  } catch (error) {
    ElMessage.error('问答服务连接失败，请检查后端是否启动')
    // 移除失败的 AI 消息
    messages.value.pop()
    // 添加错误消息
    messages.value.push({
      role: 'assistant',
      content: '抱歉，问答服务暂时不可用。请确保后端服务已启动（python backend/project.py）。',
      time: getCurrentTime()
    })
  }
}

// ==========================================
// 键盘事件处理
// ==========================================

/**
 * 处理键盘事件
 * - Enter: 发送消息
 * - Shift+Enter: 换行
 */
const handleKeydown = (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendMessage()
  }
}

// ==========================================
// 上传处理
// ==========================================

/**
 * 上传前校验
 * - 检查文件类型（仅允许 PDF 和 Word）
 * - 检查文件大小（最大 50MB）
 */
const beforeUpload = (file) => {
  const allowedTypes = ['application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
  const maxSize = 50 * 1024 * 1024 // 50MB

  if (!allowedTypes.includes(file.type) &&
      !file.name.endsWith('.pdf') &&
      !file.name.endsWith('.docx')) {
    ElMessage.error('仅支持 PDF 和 Word 文档格式')
    return false
  }

  if (file.size > maxSize) {
    ElMessage.error('文件大小不能超过 50MB')
    return false
  }

  return true
}

/**
 * 上传成功回调
 */
const handleUploadSuccess = (response) => {
  if (response.status === 'ok') {
    ElMessage.success(
      `文档上传成功！共解析 ${response.total_chunks} 个文本块`
    )
  } else {
    ElMessage.warning('文档上传完成，但解析过程中有文件失败')
  }
}

/**
 * 上传失败回调
 */
const handleUploadError = () => {
  ElMessage.error('文档上传失败，请检查后端服务是否正常运行')
}

// ==========================================
// 工具方法
// ==========================================

/**
 * 获取当前时间的格式化字符串
 * 格式: HH:MM:SS
 */
const getCurrentTime = () => {
  const now = new Date()
  return now.toLocaleTimeString('zh-CN', { hour12: false })
}

/**
 * 清空所有对话消息
 */
const clearChat = () => {
  messages.value = []
  ElMessage.success('对话已清空')
}
</script>

<style scoped>
/* ==========================================
   全局布局
   ========================================== */
.chat-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB',
    'Microsoft YaHei', sans-serif;
}

/* ==========================================
   顶部导航栏
   ========================================== */
.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  background: linear-gradient(135deg, #409eff 0%, #337ecc 100%);
  color: white;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
  z-index: 10;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  font-size: 24px;
}

.chat-header h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  letter-spacing: 1px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* ==========================================
   聊天消息区域
   ========================================== */
.chat-main {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 自定义滚动条 */
.chat-main::-webkit-scrollbar {
  width: 6px;
}
.chat-main::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 3px;
}
.chat-main::-webkit-scrollbar-track {
  background: transparent;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
}

.empty-icon {
  font-size: 64px;
  color: #c0c4cc;
  margin-bottom: 16px;
}

.empty-state p {
  margin: 4px 0;
  font-size: 16px;
}

.empty-hint {
  font-size: 14px;
  color: #a8abb2;
}

/* ==========================================
   消息气泡
   ========================================== */
.message-wrapper {
  display: flex;
  gap: 12px;
  max-width: 85%;
  animation: fadeInUp 0.3s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message-wrapper.user-message {
  margin-left: auto;
  flex-direction: row-reverse;
}

.message-wrapper.ai-message {
  margin-right: auto;
}

.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 20px;
  color: white;
}

.user-avatar {
  background: linear-gradient(135deg, #e6a23c, #d48806);
}

.ai-avatar {
  background: linear-gradient(135deg, #409eff, #337ecc);
}

.bubble {
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.6;
  font-size: 14px;
  word-break: break-word;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.user-bubble {
  background: #409eff;
  color: white;
  border-top-right-radius: 4px;
}

.ai-bubble {
  background: white;
  color: #303133;
  border-top-left-radius: 4px;
}

/* 引用来源样式 */
.references {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #ebeef5;
}

.ref-title {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #909399;
  margin-bottom: 8px;
}

.ref-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
  font-size: 12px;
}

.ref-score {
  color: #606266;
}

/* 响应时间 */
.response-time {
  margin-top: 8px;
  font-size: 11px;
  color: #c0c4cc;
}

/* 消息时间 */
.message-time {
  margin-top: 6px;
  font-size: 11px;
  opacity: 0.7;
}

.user-bubble .message-time {
  color: rgba(255, 255, 255, 0.8);
}

/* 加载中指示器 */
.loading-bubble {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 120px;
}

.typing-dots {
  display: flex;
  gap: 4px;
}

.typing-dots span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #409eff;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: 0s; }
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
  0%, 60%, 100% { transform: scale(0.6); opacity: 0.4; }
  30% { transform: scale(1); opacity: 1; }
}

/* ==========================================
   底部输入区域
   ========================================== */
.chat-footer {
  padding: 16px 24px;
  background: white;
  border-top: 1px solid #ebeef5;
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.upload-area {
  padding-top: 8px;
}

.input-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.input-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.left-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.char-count {
  font-size: 12px;
  color: #909399;
}

/* 响应式适配 */
@media (max-width: 768px) {
  .chat-header h1 {
    font-size: 16px;
  }

  .chat-main {
    padding: 16px;
  }

  .message-wrapper {
    max-width: 95%;
  }
}
</style>
