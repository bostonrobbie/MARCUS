
import { EventBus, EVENTS } from '../core/events';
import { JobQueue } from '../core/queue';
import { Logger } from '../core/logger';
import { TechnicalAnalyst } from '../quant_lab/technical_analysis';

async function main() {
    console.log('--- Verifying Backend Upgrades ---');

    const logger = new Logger();
    await logger.info('Test Log from Verification Script');

    // 1. Verify Event Bus
    console.log('1. Testing Event Bus...');
    let eventReceived = false;
    EventBus.getInstance().on('TEST_EVENT', (data) => {
        console.log(`[EventBus] Received: ${JSON.stringify(data)}`);
        eventReceived = true;
    });
    EventBus.getInstance().emit('TEST_EVENT', { msg: 'Hello EventBus' });

    // 2. Verify Job Queue
    console.log('2. Testing Job Queue...');
    console.log('2. Testing Job Queue...');
    let jobProcessed = false;
    JobQueue.register('TEST_JOB', async (payload: any) => {
        console.log(`[JobQueue] Processing: ${JSON.stringify(payload)}`);
        jobProcessed = true;
    });

    // Start worker to process it
    JobQueue.worker();

    await JobQueue.add('TEST_JOB', { foo: 'bar' });

    // 3. Verify Quant Lab Isolation
    console.log('3. Testing Quant Lab...');
    // TechnicalAnalyst contains static methods
    const rsi = TechnicalAnalyst.calculateRSI([10, 12, 11, 14, 15, 16], 3);
    console.log('Quant Lab RSI calculation: ' + rsi);
    console.log('Quant Lab (TechnicalAnalyst) loaded successfully.');

    // Wait for async processing
    setTimeout(() => {
        if (eventReceived && jobProcessed) {
            console.log('\n✅ VERIFICATION SUCCESSFUL');
            process.exit(0);
        } else {
            console.error('\n❌ VERIFICATION FAILED');
            console.error(`Event: ${eventReceived}, Job: ${jobProcessed}`);
            process.exit(1);
        }
    }, 2000);

    // Clean exit
    setTimeout(() => {
        if (!jobProcessed) {
            // Stop the loop if needed? 
            // JobQueue.stop(); 
        }
    }, 2500);
}

main().catch(console.error);
