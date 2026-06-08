/**
 * RAG 知识库问答系统 - Vue3 主入口
 *
 * 功能：
 * 1. 创建 Vue3 应用实例
 * 2. 注册 Element Plus 组件库
 * 3. 挂载根组件 App.vue
 */

import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import App from './App.vue'

// 创建 Vue 应用实例
const app = createApp(App)

// 注册 Element Plus 组件库（UI 框架）
app.use(ElementPlus, { size: 'default' })

// 全局注册所有 Element Plus 图标组件
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 挂载到 DOM
app.mount('#app')
