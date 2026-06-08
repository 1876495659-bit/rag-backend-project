<!-- RAG Knowledge Base Chat System - Vue 3 Main Entry -->
<template>
  <div class="chat-container">
    <el-header>
      <h2>Knowledge Base Chat System</h2>
      <span class="status">{{ status }}</span>
    </el-header>

    <el-main>
      <div class="messages" ref="messageContainer">
        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          :class="['message', msg.role]"
        >
          <div class="avatar">
            <el-avatar :size="40">
              <img v-if="msg.role === 'user'" src="https://element-plus.org/images/demo/ball-orange.png" alt="user">
              <img v-else src="https://element-plus.org/images/demo/ball-blue.png" alt="ai">
            </el-avatar>
          </div>
          <div class="content">
            <div class="text">{{ msg.content }}</div>
          </div>
        </div>
      </div>

      <el-footer>
        <el-input
          v-model="query"
          placeholder="Enter your question..."
          @keyup.enter="sendMessage"
          :disabled="loading"
        />
        <el-button type="primary" @click="sendMessage" :loading="loading">
          {{ loading ? 'Generating...' : 'Send' }}
        </el-button>
      </el-footer>
    </el-main>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { ElMessage } from 'element-plus'

const query = ref('')
const messages = ref([
  { role: 'assistant', content: 'Hello! How can I help you today?' }
])
const loading = ref(false)
const messageContainer = ref(null)

const sendMessage = async () => {
  if (!query.value.trim()) return

  messages.value.push({ role: 'user', content: query.value })
  query.value = ''
  loading.value = true

  try {
    const response = await fetch(`http://localhost:8000/stream/${encodeURIComponent(query.value)}`)
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let assistantMessage = ''

    messages.value.push({ role: 'assistant', content: '' })

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const text = decoder.decode(value)
      const lines = text.split('\n').filter(line => line.trim())
      for (const line of lines) {
        if (line.startsWith('data:')) {
          const chunk = line.substring(5).trim()
          assistantMessage += chunk
        }
      }
      messages.value[messages.value.length - 1].content = assistantMessage
      await nextTick()
    }
  } catch (err) {
    ElMessage.error('Request failed. Please check if the backend is running.')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #f7f8fc;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: #409eff;
  color: #fff;
}

.status {
  background: #67c23a;
  padding: 2px 12px;
  border-radius: 10px;
  font-size: 0.8rem;
}

.messages {
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.message {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
}

.message.user {
  flex-direction: row-reverse;
}

.avatar {
  margin-top: 0.2rem;
}

.content {
  background: #fff;
  padding: 1rem;
  border-radius: 1rem;
  max-width: 80%;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.message.user .content {
  background: #409eff;
  color: #fff;
}

.message.assistant .content {
  border-top-left-radius: 0.5rem;
}

.footer {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.5rem;
  background: #fff;
  box-shadow: 0 -2px 12px rgba(0, 0, 0, 0.1);
}

.loading {
  opacity: 0.7;
  cursor: not-allowed;
}

.el-avatar {
  border: 2px solid #409eff;
}

@keyframes pulse {
  0% { transform: scale(0.95); }
  50% { transform: scale(1.05); }
  100% { transform: scale(0.95); }
}

.message.loading {
  animation: pulse 1.5s infinite;
}
</style>
