
import { DB } from '../core/db';

async function main() {
    await DB.getInstance().init();
    const db = DB.getInstance().getDb();

    db.all("SELECT id, step_id, status, error, updated_at FROM process_steps WHERE process_run_id = 10", (err, rows) => {
        if (err) console.error(err);
        else console.table(rows);
    });
}

main();
