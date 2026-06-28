(function(){let e=document.createElement(`link`).relList;if(e&&e.supports&&e.supports(`modulepreload`))return;for(let e of document.querySelectorAll(`link[rel="modulepreload"]`))n(e);new MutationObserver(e=>{for(let t of e)if(t.type===`childList`)for(let e of t.addedNodes)e.tagName===`LINK`&&e.rel===`modulepreload`&&n(e)}).observe(document,{childList:!0,subtree:!0});function t(e){let t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),e.crossOrigin===`use-credentials`?t.credentials=`include`:e.crossOrigin===`anonymous`?t.credentials=`omit`:t.credentials=`same-origin`,t}function n(e){if(e.ep)return;e.ep=!0;let n=t(e);fetch(e.href,n)}})();var e={slug:`anime-lora-pipeline`,title:`Anime Lora Pipeline`,repoUrl:`https://github.com/Justin21523/anime-lora-pipeline`,readmeUrl:`https://github.com/Justin21523/anime-lora-pipeline#readme`,domain:`ai-workflow`,summary:`Anime Lora Pipeline is a curated project from the Justin21523 GitHub portfolio. A modular and fully-automated pipeline for training high-quality anime character LoRA models with Stable Diffusion. Features include video frame extraction, image cleaning, character filtering, automatic captioning, and training orchestration. Built for scalability across multiple characters and projects.`,problem:`The project needed a consistent public portfolio entry that connects the GitHub repo, README, demo status, and project metadata.`,solution:`The portfolio entry is generated from repo metadata, README content, and local source scan signals. Detected stack: Python, PyTorch, pandas.`,architecture:`This case study is generated from the portfolio catalog pipeline using README, Git metadata, package/build configuration, and media signals. The final architecture narrative still needs source-level review. Current detected technology signals include: Python, PyTorch, pandas.`,features:[`Detected technical signals: Python, PyTorch, pandas,README evidence exists and can support a fuller reviewed case study,A public GitHub repository is linked for source traceability`,`Anime LoRA Pipeline`,`📂 Project Structure`,`Install PyTorch (CUDA 12`,`Install project dependencies`,`Prepare Training Data`,`The portfolio entry is generated from repo metadata, README content, and local source scan signals`,`Detected stack: Python, PyTorch, pandas`],modules:[`.gitattributes`,`.gitignore`,`GLOSSARY.md`,`IMPLEMENTATION_SUMMARY.md`,`PHASE2_7_CLEANUP_REPORT.md`,`PHASE2_VERIFICATION_REPORT.md`,`README.md`,`README_NEW.md`,`REORGANIZATION_STATUS.md`,`U2NET_OPTIMIZATION_REPORT.md`,`USAGE_GUIDE.md`,`WAREHOUSE_SETUP.md`,`WINDOWS_SETUP_SIMPLE.txt`,`WSL_LONG_RUNNING_GUIDE.md`,`YOKAI_WATCH_PIPELINE_GUIDE.md`,`config/`,`configs/`,`docs/`],headings:[`Anime LoRA Pipeline`,`✨ Features`,`📂 Project Structure`,`🚀 Quick Start`,`1. Setup Environment`,`Install PyTorch (CUDA 12.8 for RTX 5080)`,`Install project dependencies`,`2. Prepare Training Data`,`3. Train LoRA`,`📊 Evaluation`],sampleRows:[{id:`S01`,label:`Detected technical signals: Python, PyTorch, pandas,README evidence exists and can support a fuller reviewed case study,A public GitHub repository is linked for source traceability`,value:42,delta:`+12%`,status:`ready`},{id:`S02`,label:`Anime LoRA Pipeline`,value:51,delta:`+7%`,status:`review`},{id:`S03`,label:`📂 Project Structure`,value:60,delta:`+12%`,status:`complete`},{id:`S04`,label:`Install PyTorch (CUDA 12`,value:69,delta:`+7%`,status:`ready`},{id:`S05`,label:`Install project dependencies`,value:78,delta:`+12%`,status:`review`},{id:`S06`,label:`Prepare Training Data`,value:87,delta:`+7%`,status:`complete`}],scenarios:[{id:`01`,name:`Input assets`,status:`Ready`,detail:`Detected technical signals: Python, PyTorch, pandas,README evidence exists and can support a fuller reviewed case study,A public GitHub repository is linked for source traceability`},{id:`02`,name:`Preprocessing`,status:`Ready`,detail:`Anime LoRA Pipeline`},{id:`03`,name:`Model pass`,status:`Ready`,detail:`📂 Project Structure`},{id:`04`,name:`Evaluation`,status:`Review`,detail:`Install PyTorch (CUDA 12`},{id:`05`,name:`Export`,status:`Review`,detail:`Install project dependencies`}]},t=`workspace`,n=e.scenarios[0]?.id??`01`;function r(e,t){return`<div class="metric"><span>${e}</span><strong>${t}</strong></div>`}function i(){return[`workspace`,`workflow`,`visualization`,`evidence`,`architecture`].map(e=>`<button class="${t===e?`active`:``}" data-view="${e}">${e}</button>`).join(``)}function a(){return`
    <section class="hero">
      <div>
        <p class="eyebrow">${e.domain.replaceAll(`-`,` `)}</p>
        <h1>${e.title}</h1>
        <p class="lead">${e.summary}</p>
      </div>
      <div class="metrics">
        ${r(`Workflow steps`,e.scenarios.length)}
        ${r(`Source modules`,e.modules.length)}
        ${r(`Review mode`,`Static`)}
        ${r(`Backend`,`Fixture`)}
      </div>
    </section>
    <section class="split">
      <article><h2>Problem</h2><p>${e.problem}</p></article>
      <article><h2>Implemented Result</h2><p>${e.solution}</p></article>
    </section>
  `}function o(){return`
    <section>
      <div class="section-head">
        <div><p class="eyebrow">project workflow</p><h2>Executable Review Path</h2></div>
        <button class="primary" id="run">Run workflow</button>
      </div>
      <div class="board">
        ${e.scenarios.map(e=>`
          <button class="card ${n===e.id?`selected`:``}" data-step="${e.id}">
            <span>${e.id}</span>
            <strong>${e.name}</strong>
            <em>${e.status}</em>
            <p>${e.detail}</p>
          </button>
        `).join(``)}
      </div>
      <output id="output">Select a step or run the workflow to inspect deterministic project output.</output>
    </section>
  `}function s(){let t=Math.max(...e.sampleRows.map(e=>e.value),1);return`
    <section>
      <div class="section-head">
        <div><p class="eyebrow">sample data result</p><h2>Visible Demo Output</h2></div>
        <span class="badge">Fixture-backed</span>
      </div>
      <div class="viz">
        <div class="bars">
          ${e.sampleRows.map(e=>`
            <div class="bar-row">
              <span>${e.id}</span>
              <div class="bar-track"><div class="bar-fill" style="width: ${Math.round(e.value/t*100)}%"></div></div>
              <strong>${e.value}</strong>
            </div>
          `).join(``)}
        </div>
        <div class="result-table">
          ${e.sampleRows.map(e=>`
            <article>
              <b>${e.label}</b>
              <span>${e.status}</span>
              <em>${e.delta}</em>
            </article>
          `).join(``)}
        </div>
      </div>
    </section>
  `}function c(){return`
    <section class="split">
      <article>
        <p class="eyebrow">repository evidence</p>
        <h2>Source modules</h2>
        <div class="chips">${e.modules.map(e=>`<span>${e}</span>`).join(``)||`<span>Project source reviewed</span>`}</div>
      </article>
      <article>
        <p class="eyebrow">documentation</p>
        <h2>README signals</h2>
        <ul>${e.headings.map(e=>`<li>${e}</li>`).join(``)||`<li>README content is represented in the workflow panels.</li>`}</ul>
      </article>
    </section>
  `}function l(){return`
    <section>
      <p class="eyebrow">architecture</p>
      <h2>Static deployment architecture</h2>
      <p>${e.architecture}</p>
      <pre>npm run dev
npm run build
GitHub Pages / gh-pages
local fixtures / no private backend</pre>
    </section>
  `}function u(){let r={workspace:a,workflow:o,visualization:s,evidence:c,architecture:l};document.querySelector(`#app`).innerHTML=`
    <header>
      <a href="${e.repoUrl}" class="brand">${e.title}</a>
      <nav>${i()}</nav>
      <a class="readme" href="${e.readmeUrl}">README</a>
    </header>
    <main>${r[t]()}</main>
  `,document.querySelectorAll(`[data-view]`).forEach(e=>e.addEventListener(`click`,()=>{t=e.dataset.view,u()})),document.querySelectorAll(`[data-step]`).forEach(e=>e.addEventListener(`click`,()=>{n=e.dataset.step,u()})),document.querySelector(`#run`)?.addEventListener(`click`,()=>{let t=document.querySelector(`#output`);t&&(t.textContent=`${e.title}: ${e.scenarios.length} workflow steps completed using local fixture state.`)})}u();