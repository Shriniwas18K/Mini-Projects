import { FC } from "react";
import Card from "antd/es/card/Card";
import Meta from "antd/es/card/Meta";
import { Avatar } from "antd";

export interface Student {
  name: string;
  attendance: number;
}

export const StudentCard: FC<Student> = (student) => {
  return (
    <>
      <Card
        style={{
          maxWidth: "10cm",
          minWidth: "7cm",
          minHeight: "5cm",
          background: "rgba(255, 255, 255, 0.4)",
          borderRadius: "16px",
          boxShadow: "0 4px 30px rgba(0, 0, 0, 0.1)",
          border: "1px solid rgba(255, 255, 255, 0.5)",
        }}
      >
        <Meta
          avatar={
            <Avatar
              src={`https://api.dicebear.com/7.x/miniavs/svg`}
              alt="img"
            />
          }
          title={student.name}
          description={student.attendance}
        />
      </Card>
    </>
  );
};
