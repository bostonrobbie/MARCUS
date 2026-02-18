import { EventEmitter } from 'events';

export class EventBus extends EventEmitter {
    private static instance: EventBus;

    private constructor() {
        super();
        this.setMaxListeners(20);
    }

    public static getInstance(): EventBus {
        if (!EventBus.instance) {
            EventBus.instance = new EventBus();
        }
        return EventBus.instance;
    }
}

export const EVENTS = {
    TASK: {
        CREATED: 'task:created',
        UPDATED: 'task:updated',
    },
    SCHEDULE: {
        DUE: 'schedule:due',
    },
    SYSTEM: {
        TICK: 'system:tick'
    }
};
