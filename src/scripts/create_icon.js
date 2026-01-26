const fs = require('fs');
const pngToIco = require('png-to-ico');
const path = require('path');

// I need to copy the artifact to the current directory first, but I don't know the exact name.
// Wait, the tool output gave me the path. 
// "C:/Users/User/.gemini/antigravity/brain/f52e9b8b-6b7d-4e68-81e0-b2ed2e916027/hedge_fund_ai_logo_1769365201119.png"

const sourcePath = 'temp_logo.png';
const destDir = path.join(__dirname, '../dashboard');
const destFile = path.join(destDir, 'favicon.ico');

if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir, { recursive: true });
}

// Fix for default export
const convert = pngToIco.default;

convert(sourcePath)
    .then(buf => {
        fs.writeFileSync(destFile, buf);
        console.log('Successfully created favicon.ico at ' + destFile);

        // Also save a copy to root for the user to use on Desktop
        fs.writeFileSync('C:/Users/User/Documents/AI/local-manus-agent-workspace/ai-company-os/app_icon.ico', buf);
        console.log('Successfully created app_icon.ico in root');
    })
    .catch(console.error);
