import fs from 'fs';
import MarkdownIt from 'markdown-it';
import { google } from 'googleapis';

async function main() {
  const md = new MarkdownIt();
  const html = md.render(fs.readFileSync('post.md', 'utf8'));

  const auth = new google.auth.OAuth2(
    process.env.CLIENT_ID,
    process.env.CLIENT_SECRET
  );
  auth.setCredentials({ refresh_token: process.env.REFRESH_TOKEN });

  const blogger = google.blogger({ version: 'v3', auth });

  await blogger.posts.insert({
    blogId: process.env.BLOG_ID,
    requestBody: {
      title: 'GitHub에서 자동 게시된 글',
      content: html
    }
  });

  console.log('✅ 블로그에 글이 올라갔어요!');
}

main().catch(console.error);
