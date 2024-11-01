import { useState } from "react";
import { StudentCard } from "./components/StudentCard";
import { Flex, Layout, Input, Radio, RadioChangeEvent } from "antd";
import { Student } from "./components/StudentCard";

const { Search } = Input;
const { Header } = Layout;

const studentsData: Student[] = [
  { name: "ABC PQR", attendance: 89.9 },
  { name: "HGE WWERT", attendance: 99.9 },
  { name: "SHRI TWT", attendance: 90.9 },
  { name: "SDGWE RWE", attendance: 75.9 },
  { name: "TRYH EYE55", attendance: 70.9 },
  { name: "SHRI", attendance: 39.9 },
  { name: "OFERTRW", attendance: 91.9 },
  { name: "MOWETOT", attendance: 92.23 },
];

const attendanceThresholds = {
  active: 75,
  defaulters: 50,
};

function App() {
  const [status, setStatus] = useState("all");
  const [searchText, setSearchText] = useState("");

  const handleRadioChange = (e: RadioChangeEvent) => {
    setStatus(e.target.value);
  };

  const handleSearch = (value: string) => {
    setSearchText(value);
  };

  const filteredStudents = studentsData
    .filter((student) => {
      if (status === "all") return true;
      if (status === "active")
        return student.attendance >= attendanceThresholds.active;
      if (status === "defaulters")
        return student.attendance < attendanceThresholds.defaulters;
      return true;
    })
    .filter((student) =>
      student.name.toLowerCase().includes(searchText.toLowerCase())
    );

  return (
    <Layout
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(to right top, rgb(249, 168, 212), rgb(255, 200, 254), rgb(129, 140, 248))",
      }}
    >
      <Header
        style={{
          background: "rgba(0,0,0,0)",
          padding: "10vh",
          display: "flex",
          flexFlow: "row wrap",
          justifyContent: "space-between",
        }}
      >
        <Search
          placeholder="input search text"
          style={{ maxWidth: "10cm" }}
          allowClear
          onChange={(e) => handleSearch(e.target.value)}
        />
        <Radio.Group onChange={handleRadioChange} style={{ marginBottom: 16 }}>
          <Radio.Button value="all">All</Radio.Button>
          <Radio.Button value="active">Active</Radio.Button>
          <Radio.Button value="defaulters">Defaulters</Radio.Button>
        </Radio.Group>
      </Header>
      <Flex
        wrap
        gap={"large"}
        justify={"center"}
        align={"center"}
        style={{ padding: "10vh" }}
      >
        {filteredStudents.map((student, index) => (
          <StudentCard
            key={index}
            name={student.name}
            attendance={student.attendance}
          />
        ))}
      </Flex>
    </Layout>
  );
}

export default App;
