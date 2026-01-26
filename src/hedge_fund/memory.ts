
import { DB } from '../core/db';

export class MemoryStore {
    public static async add(projectId: number, dept: string, title: string, content: string, tags: string = '', timelineIndex: number = 0) {
        const prisma = DB.getInstance().getPrisma();
        const memory = await prisma.memory.create({
            data: {
                projectId: projectId,
                dept: dept,
                title: title,
                content: content,
                tags: tags,
                timelineIndex: timelineIndex,
                createdAt: new Date()
            }
        });
        return memory.id;
    }

    public static async search(projectId: number, dept: string, query: string): Promise<any[]> {
        const prisma = DB.getInstance().getPrisma();
        // Prisma doesn't have good 'OR' with multiple fields LIKE ...
        // We can use explicit ORs
        const containing = { contains: query };
        return prisma.memory.findMany({
            where: {
                projectId: projectId,
                dept: { in: [dept, 'system'] },
                OR: [
                    { title: containing },
                    { content: containing },
                    { tags: containing }
                ]
            },
            orderBy: [
                { timelineIndex: 'desc' },
                { createdAt: 'desc' }
            ],
            take: 5
        });
    }

    public static async getTimeline(projectId: number, limit: number = 20): Promise<any[]> {
        const prisma = DB.getInstance().getPrisma();
        return prisma.memory.findMany({
            where: { projectId: projectId },
            orderBy: [
                { timelineIndex: 'asc' },
                { createdAt: 'asc' }
            ],
            take: limit
        });
    }
}
