# 🚀 Initio Lily - Future Roadmap & Feature Backlog

เอกสารรวบรวมแผนการพัฒนาฟีเจอร์และแนวคิดสำหรับต่อยอด **Lily AI Assistant** ในอนาคต

---

## 📌 Feature Backlog

### 1. 🧠 Model Introspection & VRAM Telemetry (`/model` Command)
- **คำอธิบาย**: เพิ่มความสามารถให้ Agent ตรวจสอบสเปกและสถานะของตัวเองผ่าน Ollama SDK
- **ฟีเจอร์หลัก**:
  - ดึงข้อมูลสถาปัตยกรรมโมเดล, ขนาดพารามิเตอร์ (เช่น 32B), Quantization level (เช่น BF16, Q4_K_M) ผ่าน `client.show(model_name)`
  - แสดงสถานะการรันบน GPU/VRAM และ RAM แบบ Real-time ผ่าน `client.ps()`
  - เพิ่มคำสั่งลัด **`/model`** ใน Terminal CLI เพื่อดูข้อมูลสรุปของ Model ปัจจุบัน

---

### 2. 📝 Local Notes Tool (Knowledge Base & Documentation) `[✔ Completed]`
- **คำอธิบาย**: เครื่องมือจัดการไฟล์บันทึก Markdown (`.md`) ในเครื่องคอมพิวเตอร์ของผู้ใช้โดยตรง
- **ฟีเจอร์หลัก**:
  - `create_or_update_note(filename, content)`: สรุปและบันทึกเนื้อหาลงโฟลเดอร์ `notes/`
  - `read_note(filename)`: อ่านไฟล์บันทึกเดิมมาวิเคราะห์หรือตอบคำถาม
  - `append_to_note(filename, content)`: บันทึกข้อมูลต่อท้าย (เช่น Daily logs, Ideas)
  - `list_notes()`: แสดงรายชื่อโน้ตทั้งหมด
  - `search_notes(query)`: ค้นหาคีย์เวิร์ดในเนื้อหาโน้ตทั้งหมด (Full-text Keyword Search)
  - คำสั่งลัด `/notes` บน Rich Terminal CLI

---

### 3. 🗂️ Multi-Session Management (Session Switcher)
- **คำอธิบาย**: ยกระดับระบบ Session บน PostgreSQL ให้สลับห้องคุยได้อิสระ
- **ฟีเจอร์หลัก**:
  - `/new [title]`: สร้างเซสชันการสนทนาใหม่
  - `/sessions`: แสดงรายการประวัติห้องสนทนาทั้งหมดพร้อมวันที่คุยล่าสุด
  - `/switch <session_id>`: สลับไปคุยต่อในเซสชันเดิม
  - `/rename <title>`: เปลี่ยนชื่อหัวข้อเซสชัน

---

### 4. 📱 Mobile Messaging Bridge (Telegram / Discord Bot)
- **คำอธิบาย**: เชื่อมต่อ Lily เข้ากับแพลตฟอร์มแชต เพื่อให้ใช้งานผ่านสมาร์ตโฟนได้ทุกที่ทุกเวลา
- **ฟีเจอร์หลัก**:
  - สร้าง Telegram Bot Gateway ด้วย `python-telegram-bot` หรือ FastAPI Webhook
  - ใช้งาน Core ReAct Engine, Todoist Tools, Memory, และ Web Search ร่วมกันกับ CLI 100%

---

### 5. 🔍 Semantic Memory & Vector Search (pgvector Integration)
- **คำอธิบาย**: ค้นหาความจำและความรู้ย้อนหลังตามความหมายเชิงลึก (Semantic Recall)
- **ฟีเจอร์หลัก**:
  - ใช้ Extension `pgvector` ที่เตรียมไว้ใน Docker PostgreSQL
  - แปลงข้อความประวัติเก่าเป็น Embeddings เพื่อดึงบริบทที่เกี่ยวข้องแม้จะไม่ได้ใช้คีย์เวิร์ดตรงตัว

---

### 6. ⏰ Proactive & Scheduled Tasks (Background Assistant)
- **คำอธิบาย**: ให้ Lily สามารถแจ้งเตือนหรือสรุปงานให้ผู้ใช้เองโดยไม่ต้องรอให้สั่ง
- **ฟีเจอร์หลัก**:
  - Daily Morning Briefing: สรุปงานค้างใน Todoist และสภาพอากาศตอน 08:00 น.
  - Proactive Reminders: แจ้งเตือนเมื่อใกล้ถึงเวลานัดหมาย

---

### 7. 📚 Advanced Output Pagination / RAG for Large Files
- **คำอธิบาย**: ปรับปรุงการอ่านไฟล์ขนาดใหญ่มากๆ (เช่น PDF หลักร้อยหน้า หรือ CSV หมื่นบรรทัด)
- **ฟีเจอร์หลัก**:
  - สร้างระบบ Chunking หรือ Pagination ใน `read_document` Tool แทนการ Truncate (ตัดทิ้ง) แบบทื่อๆ
  - ให้ Agent ใช้คำสั่ง `read_next_page()` หรือประยุกต์ใช้ Local RAG เพื่อดึงเฉพาะบริบทที่สำคัญมาตอบ

---

### 8. 🤖 LLM-Based Dynamic Tool Router (Semantic Tool Selection)
- **คำอธิบาย**: ยกระดับระบบ Dynamic Tool Filtering จากเดิมที่ใช้ Keyword/Regex ไปเป็น **LLM-Based / Semantic Router** เพื่อวิเคราะห์ความตั้งใจของผู้ใช้ได้อย่างแม่นยำยิ่งขึ้น
- **ฟีเจอร์หลัก**:
  - ใช้ Fast Small Model (SLM ขนาด 1B-3B) หรือ Embedding Vector Similarity ในการทำนายและคัดเลือก Tools ที่จำเป็น 3–5 ตัวก่อนส่งเข้า Main LLM
  - รองรับคำสั่งที่มีความหมายแฝง (Implicit Intents) หรือการอธิบายแบบเปรียบเปรยที่ไม่มีคีย์เวิร์ดตรงตัว
  - สลับและจัดกลุ่มชุดเครื่องมือ (Dynamic Tool Binding) ได้อย่างชาญฉลาดโดยไม่เกิดปัญหา Tool Schema Overload
