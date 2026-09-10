import {execFileSync} from 'node:child_process';
import {readFileSync, existsSync} from 'node:fs';
import path from 'node:path';
const files = execFileSync('git', ['ls-files', '-z'], {encoding:'utf8'}).split('\0').filter(Boolean);
const failures = [];
let documents = 0;
for (const file of files) {
  if (file.endsWith('.json')) {
    try { JSON.parse(readFileSync(file, 'utf8')); } catch (error) { failures.push(`${file}: invalid JSON: ${error.message}`); }
  }
  if (!file.endsWith('.md')) continue;
  documents++;
  const text = readFileSync(file, 'utf8').replace(/^\s*(```|~~~)[\s\S]*?^\s*\1.*$/gm, '');
  for (const match of text.matchAll(/\[[^\]\n]*\]\(([^)\n]+)\)/g)) {
    const target = match[1].trim().replace(/^<([^>]+)>.*$/, '$1').replace(/\s+["'][\s\S]*$/, '');
    if (!target || /^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(target)) continue;
    let local;
    try { local = decodeURIComponent(target.split(/[?#]/)[0]); } catch { failures.push(`${file}: malformed link ${target}`); continue; }
    const resolved = local.startsWith('/') ? path.resolve('.'+local) : path.resolve(path.dirname(file),local);
    if (!existsSync(resolved)) failures.push(`${file}: missing local link target ${target}`);
  }
}
if (failures.length) { console.error(failures.join('\n')); process.exit(1); }
console.log(`Validated local links in ${documents} Markdown documents and tracked JSON syntax.`);
