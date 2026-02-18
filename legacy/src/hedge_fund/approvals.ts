
import { DB } from '../core/db';
import { Task } from '../quant_lab/types';

// Add ValidatedApproval to types.ts first or define inline? 
// I'll assume I update types.ts later or just use what I have.
// Wait, I didn't verify ValidatedApproval in types.ts. It's not there.
// I'll stick to the table structure 'approvals'.

export class ApprovalSystem {

    public static async createRequest(taskId: number, notes: string): Promise<number> {
        const prisma = DB.getInstance().getPrisma();

        const approval = await prisma.approval.create({
            data: {
                taskId: taskId,
                requestedAt: new Date(),
                decision: 'PENDING',
                notes: notes
            }
        });
        return approval.id;
    }

    public static async checkStatus(approvalId: number): Promise<'APPROVED' | 'REJECTED' | 'PENDING'> {
        const prisma = DB.getInstance().getPrisma();
        const approval = await prisma.approval.findUnique({
            where: { id: approvalId },
            select: { decision: true }
        });

        if (!approval) throw new Error('Approval not found');
        return approval.decision as 'APPROVED' | 'REJECTED' | 'PENDING';
    }

    public static async review(approvalId: number, decision: 'APPROVED' | 'REJECTED', notes?: string): Promise<void> {
        const prisma = DB.getInstance().getPrisma();
        await prisma.approval.update({
            where: { id: approvalId },
            data: {
                decision: decision,
                approvedAt: new Date(),
                notes: notes || undefined // Prisma doesn't support COALESCE directly but user passes notes? Logic was COALESCE(?, notes), meaning if param is null keep existing? Prisma update replaces fields. 
                // Wait, if notes is optional in update, undefined means "do not touch".
                // original sql: notes = COALESCE(?, notes). ? is the notes arg.
                // If notes arg is null/undefined, it keeps notes.
                // So passing undefined to prisma means "do not update".
            }
        });
    }

    public static async isApproved(taskId: number): Promise<boolean> {
        const prisma = DB.getInstance().getPrisma();
        const count = await prisma.approval.count({
            where: {
                taskId: taskId,
                decision: 'APPROVED'
            }
        });
        return count > 0;
    }

    public static async getPending(): Promise<any[]> {
        const prisma = DB.getInstance().getPrisma();
        const approvals = await prisma.approval.findMany({
            where: { decision: 'PENDING' },
            include: { task: true }
        });

        return approvals.map(a => ({
            id: a.id,
            task_id: a.taskId,
            requested_at: a.requestedAt,
            notes: a.notes,
            title: a.task.title
        }));
    }
}
