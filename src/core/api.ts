
import * as http from 'http';
import { DB } from './db';
import * as fs from 'fs';
import * as path from 'path';

export class ApplianceAPI {
    private server: http.Server;
    private port: number = 3000;

    constructor() {
        this.server = http.createServer(async (req, res) => {
            if (req.url === '/health' && req.method === 'GET') {
                await this.handleHealth(req, res);
            } else {
                res.writeHead(404);
                res.end();
            }
        });
    }

    private async handleHealth(req: http.IncomingMessage, res: http.ServerResponse) {
        try {
            await DB.getInstance().init();
            const prisma = DB.getInstance().getPrisma();

            // Check pending approvals
            const pending = await prisma.approval.count({
                where: { decision: 'PENDING' }
            });

            // Check heartbeat
            let heartbeat = 'UNKNOWN';
            const hbPath = path.join(process.cwd(), 'daemon_heartbeat.json');
            if (fs.existsSync(hbPath)) {
                const data = JSON.parse(fs.readFileSync(hbPath, 'utf8'));
                heartbeat = data.last_tick;
            }

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                status: 'UP',
                db: 'OK',
                pending_approvals: pending,
                scheduler_heartbeat: heartbeat,
                binding: '127.0.0.1'
            }));
        } catch (e: any) {
            res.writeHead(500);
            res.end(JSON.stringify({ status: 'DOWN', error: e.message }));
        }
    }

    public start() {
        this.server.listen(this.port, '127.0.0.1', () => {
            console.log(`[API] Appliance Health API running at http://127.0.0.1:${this.port}/health`);
        });
    }

    public stop() {
        this.server.close();
    }
}
