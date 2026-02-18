
import { DB } from '../core/db';
import { ProcessRunner } from '../hedge_fund/process_runner';

async function main() {
    await DB.getInstance().init();
    console.log("Starting proprietary_data_audit...");

    try {
        const runId = await ProcessRunner.startProcess('proprietary_data_audit', 1, {
            target_path: "C:/Users/User/Desktop/Chatgpt"
        });
        console.log(`Process started successfully. Run ID: ${runId}`);

        // Kick it off
        const runner = new ProcessRunner(0);
        await runner.executeNextStep(runId);
        console.log("First step executed.");

    } catch (e: any) {
        console.error("Failed to start:", e);
    }
}

main();
