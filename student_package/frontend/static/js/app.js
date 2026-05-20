// 多模态历史 RAG Agent — 前端逻辑
'use strict';

const API = {
  metrics: () => fetch('/api/metrics').then(r => r.json()),
  timeline: () => fetch('/api/timeline').then(r => r.json()),
  sections: () => fetch('/api/sections').then(r => r.json()),
  questions: (section = '') => fetch(`/api/questions?section=${encodeURIComponent(section)}`).then(r => r.json()),
  question: (idx) => fetch(`/api/question/${idx}`).then(r => r.json()),
  ask: (data) => fetch('/api/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data),
  }).then(r => r.json()),
};

const SECTION_ICONS = {
  '判断QA': '✓',
  '判断VQA': '✓📷',
  '简答QA': '✎',
  '简答VQA': '✎📷',
  '选择QA': '⒜',
  '选择VQA': '⒜📷',
};

// ---- tab 切换 ----
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    document.getElementById(tab).classList.add('active');
    if (tab === 'demo') ensureDemoLoaded();
  });
});

// ============== Overview ==============

(async function loadOverview() {
  const m = await API.metrics();
  document.getElementById('metric-total').textContent = `${(m.accuracy * 100).toFixed(0)}%`;
  document.getElementById('metric-correct').textContent = m.correct;
  document.getElementById('metric-all').textContent = m.total;

  const bars = document.getElementById('section-bars');
  bars.innerHTML = '';
  const sections = m.by_section || {};
  const order = ['判断QA', '判断VQA', '简答QA', '简答VQA', '选择QA', '选择VQA'];
  order.forEach(name => {
    const sec = sections[name];
    if (!sec) return;
    const pct = sec.accuracy * 100;
    const cls = pct >= 80 ? '' : (pct >= 50 ? 'partial' : 'low');
    const div = document.createElement('div');
    div.className = 'section-bar';
    div.innerHTML = `
      <div class="label">${SECTION_ICONS[name] || ''} ${name}</div>
      <div class="bar-track">
        <div class="bar-fill ${cls}" style="width: ${pct}%">
          ${pct.toFixed(0)}%
        </div>
      </div>
      <div class="meta">${sec.correct} / ${sec.total}</div>
    `;
    bars.appendChild(div);
  });
})();

// ============== Methods ==============

(async function loadTimeline() {
  const tl = await API.timeline();
  const container = document.getElementById('timeline');
  container.innerHTML = '';
  tl.forEach(step => {
    const el = document.createElement('div');
    el.className = 'timeline-step' + (step.name === 'v2' ? ' best' : '');
    const score = step.score_50 != null ? `${step.score_50}%` : '—';
    const mock = step.score_120 != null ? ` · mock ${step.score_120}%` : '';
    el.innerHTML = `
      <div>
        <div class="step-name">${step.title}</div>
        <div class="step-delta">${step.delta}${mock}</div>
      </div>
      <div class="timeline-highlights">
        <ul>${step.highlights.map(h => `<li>${h}</li>`).join('')}</ul>
      </div>
      <div style="text-align: right">
        <div class="step-score">${score}</div>
        <div class="step-delta">50 题官评</div>
      </div>
    `;
    container.appendChild(el);
  });
})();

// ============== Demo ==============

let _demoLoaded = false;
let _currentSection = '';

async function ensureDemoLoaded() {
  if (_demoLoaded) return;
  _demoLoaded = true;
  const sections = await API.sections();
  const filterEl = document.getElementById('section-filter');
  filterEl.innerHTML = '';
  const allBtn = document.createElement('button');
  allBtn.className = 'section-btn active';
  allBtn.dataset.section = '';
  allBtn.innerHTML = `<span>全部</span><span>${sections.reduce((s, x) => s + x.count, 0)}</span>`;
  allBtn.addEventListener('click', () => selectSection(''));
  filterEl.appendChild(allBtn);

  sections.forEach(s => {
    const btn = document.createElement('button');
    btn.className = 'section-btn';
    btn.dataset.section = s.section;
    btn.innerHTML = `<span>${SECTION_ICONS[s.section] || ''} ${s.section}</span><span>${s.count}</span>`;
    btn.addEventListener('click', () => selectSection(s.section));
    filterEl.appendChild(btn);
  });

  await loadQuestionList('');
}

async function selectSection(section) {
  _currentSection = section;
  document.querySelectorAll('.section-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.section === section);
  });
  await loadQuestionList(section);
}

async function loadQuestionList(section) {
  const items = await API.questions(section);
  const container = document.getElementById('question-items');
  container.innerHTML = '';
  if (items.length === 0) {
    container.innerHTML = '<div class="placeholder">该题型无题目</div>';
    return;
  }
  items.forEach(q => {
    const div = document.createElement('div');
    div.className = 'question-item';
    const icon = q.correct ? '✅' : '❌';
    div.innerHTML = `
      <span class="q-icon">${icon}</span>
      <span class="q-text" title="${escapeHtml(q.question)}">[${q.index}] ${escapeHtml(q.question.slice(0, 40))}${q.question.length > 40 ? '...' : ''}</span>
    `;
    div.addEventListener('click', () => {
      document.querySelectorAll('.question-item').forEach(x => x.classList.remove('selected'));
      div.classList.add('selected');
      showQuestion(q.index);
    });
    container.appendChild(div);
  });
}

async function showQuestion(idx) {
  const data = await API.question(idx);
  const detail = document.getElementById('question-detail');
  if (data.error) {
    detail.innerHTML = `<div class="placeholder">题目 ${idx} 不存在</div>`;
    return;
  }

  const q = data.question;
  const pred = data.prediction || {};
  const tc = data.time_cost || {};
  const correctClass = data.correct ? 'correct' : 'incorrect';
  const statusText = data.correct ? '✅ 答对' : '❌ 答错';

  let imageHtml = '';
  if (q.image_url) {
    const fname = q.image_url.split('/').pop();
    imageHtml = `<img class="qd-image" src="/images/${fname}" alt="题目图片" onerror="this.style.display='none'">`;
  }

  const logHtml = data.log_lines && data.log_lines.length
    ? data.log_lines.map(escapeHtml).join('\n')
    : '（无可用 log）';

  const expectedAnswer = q.expected_answer || '—';

  detail.innerHTML = `
    <span class="qd-section">${SECTION_ICONS[q.section] || ''} ${q.section}</span>
    <span style="float: right; color: #6b5a44; font-size: 13px;">题目 #${idx}</span>
    <div class="qd-question">${escapeHtml(q.question)}</div>
    ${imageHtml}
    <div class="qd-route">
      <strong>📍 路由方法：</strong>${escapeHtml(data.route)}
    </div>
    <h4 style="margin-top: 16px;">检索过程（来自 run.log）</h4>
    <div class="qd-log">${logHtml}</div>
    <h4 style="margin-top: 16px;">推理结果</h4>
    <div class="qd-result">
      <div class="qd-result-box ${correctClass}">
        <div class="label">模型预测</div>
        <div class="value">${escapeHtml(pred.answer || '—')}</div>
      </div>
      <div class="qd-result-box correct">
        <div class="label">正确答案</div>
        <div class="value">${escapeHtml(expectedAnswer)}</div>
      </div>
    </div>
    <div class="qd-status ${correctClass}">${statusText}</div>
    <div class="qd-meta">
      <span>⏱ 检索: ${(tc.retrieval_cost || 0).toFixed(2)}s</span>
      <span>⏱ 推理: ${(tc.infer_cost || 0).toFixed(2)}s</span>
      <span>raw: <code>${escapeHtml((pred.raw_answer || '').slice(0, 60))}</code></span>
    </div>
  `;
}

// ============== Free Ask ==============

const ROUTE_OF_SECTION_FRONTEND = {
  '判断QA': 'BM25 hybrid',
  '判断VQA': 'BM25 hybrid',
  '简答QA': 'HyDE + IB rerank',
  '简答VQA': 'vector-only',
  '选择QA': 'vector-only + choice-as-judge',
  '选择VQA': 'vector-only + choice-as-judge',
};

const askSection = document.getElementById('ask-section');
const askHint = document.getElementById('ask-route-hint');

function updateAskHint() {
  const route = ROUTE_OF_SECTION_FRONTEND[askSection.value] || 'vector-only';
  askHint.textContent = `→ 评测时该题型走 [${route}] 路由（演示模式仅闭卷调用，不带 RAG）`;
}
askSection.addEventListener('change', updateAskHint);
updateAskHint();

document.getElementById('ask-btn').addEventListener('click', async () => {
  const btn = document.getElementById('ask-btn');
  const result = document.getElementById('ask-result');
  const question = document.getElementById('ask-question').value.trim();
  const section = askSection.value;
  const imageUrl = document.getElementById('ask-image').value.trim();
  if (!question) {
    result.innerHTML = `<div class="ask-card"><div class="ask-error">题目不能为空</div></div>`;
    return;
  }
  btn.disabled = true;
  btn.textContent = '思考中...';
  result.innerHTML = `<div class="ask-card">⏳ 调用 qwen2.5-vl-7b-instruct...</div>`;
  try {
    const data = await API.ask({question, section, image_url: imageUrl});
    if (data.error) {
      result.innerHTML = `<div class="ask-card"><div class="ask-error">${escapeHtml(data.error)}</div></div>`;
    } else {
      result.innerHTML = `
        <div class="ask-card">
          <div style="font-size: 12px; color: #6b5a44;">题型: <strong>${escapeHtml(data.section)}</strong> · 路由: ${escapeHtml(data.route)} · 模型: ${escapeHtml(data.model || '')}</div>
          <div class="ask-answer">${escapeHtml(data.answer)}</div>
          <details>
            <summary style="cursor: pointer; font-size: 13px; color: #6b5a44;">查看用到的 system prompt 摘录</summary>
            <pre class="ask-prompt-preview">${escapeHtml(data.system_prompt_preview)}</pre>
          </details>
          <div style="font-size: 12px; color: #6b5a44; margin-top: 12px; font-style: italic;">${escapeHtml(data.note || '')}</div>
        </div>
      `;
    }
  } catch (e) {
    result.innerHTML = `<div class="ask-card"><div class="ask-error">网络错误: ${escapeHtml(String(e))}</div></div>`;
  }
  btn.disabled = false;
  btn.textContent = '提问';
});

// ============== utils ==============

function escapeHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
