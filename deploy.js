const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');
const { google } = require('googleapis');

const POSTS_DIR = 'content/posts/ai-basics';

async function main() {
  const md = new MarkdownIt();

  // OAuth 인증 설정
  const auth = new google.auth.OAuth2(
    process.env.CLIENT_ID,
    process.env.CLIENT_SECRET
  );
  auth.setCredentials({ refresh_token: process.env.REFRESH_TOKEN });

  const blogger = google.blogger({ version: 'v3', auth });

  // 마크다운 파일 목록 가져오기
  const files = fs.readdirSync(POSTS_DIR).filter(file => file.endsWith('.md'));

  for (const file of files) {
    const filePath = path.join(POSTS_DIR, file);
    const markdown = fs.readFileSync(filePath, 'utf8');
    const html = md.render(markdown);

    // 제목: 파일 이름에서 확장자 제거하고 숫자 앞자리 제거
    const rawTitle = path.basename(file, '.md');
    const title = rawTitle.replace(/^\d+-/, '').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

    console.log(`🚀 Posting: ${title}`);

    // Blogger에 새 글 게시
    await blogger.posts.insert({
      blogId: process.env.BLOG_ID,
      requestBody: {
        title,
        content: html,
        labels: ['AI', 'Auto-Posted']  // 선택: 라벨 지정 가능
      }
    });

    console.log(`✅ 블로그에 게시 완료: ${title}`);
  }
}

main().catch(err => {
  console.error('❌ 오류 발생:', err.message);
});
